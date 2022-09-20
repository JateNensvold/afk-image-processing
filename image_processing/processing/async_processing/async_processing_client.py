import asyncio
import json
import logging
from collections import deque
import time
from typing import Any, Callable, Coroutine, Iterable, NamedTuple
from enum import Enum
from uuid import uuid4
import concurrent.futures

import zmq
import zmq.asyncio
from zmq.log.handlers import PUBHandler
from image_processing.processing.async_processing.processing_response import (
    ProcessingResponse)

ProcessingArgs = list[str]
AsyncTask = Coroutine[Any, Any, ProcessingResponse]
# AsyncArgs = tuple[str, ProcessingArgs, int, str]
ProcessingCallback = Callable | None
# TaskArgs = tuple[AsyncTask, str, ProcessingCallback]
CallbackArgs = Any | None
QueueTimeout = int | None
TaskUUID = str

APPEND_INDEX = -1
PRIORITY_INDEX = 0


class CallbackWrapper(NamedTuple):
    """
    A wrapper around a CoRoutine callback and its arguments
    """
    callback: Coroutine
    args: Iterable


class ProcessingTaskArgs(NamedTuple):
    """
    All the args passed to the Async Compute engine
    """
    compute_task: AsyncTask
    task_uuid: TaskUUID
    callback_wrapper: CallbackWrapper | None


class ProcessingState(Enum):
    """
    All the states that a processing task can be in
    """
    not_started = 0
    in_progress = 1
    done = 2
    failed = 3


class AsyncProcessingClient:
    """
    A class that runs asynchronous image_processing by pulling tasks from a
    PriorityQueue. Tasks in the queue can be configured to timeout while
    waiting or wait until their chance for computation arrives
    """

    def __init__(self, queue_timeout: QueueTimeout = 300):
        """
        Initialize the processing client by creating a zmq socket that will be
        used to connect to the image processing host

        Args:
            queue_timeout (int, optional): a timeout used to define how long a
                task will wait in the image processing queue before timing out.
                Defaults to 300. If set to None then images in queue will wait
                indefinitely
        """

        self.queue_timeout = queue_timeout

        # zmq_log_handler = PUBHandler('tcp://127.0.0.1:12345')
        # zmq_log_handler = PUBHandler(f'tcp://{address}:{port}')

        # logger = logging.getLogger()
        # logger.addHandler(zmq_log_handler)

        self.zmq_context = zmq.asyncio.Context()
        # REQ socket can only send a single message a time
        self.zmq_socket: zmq.Socket = self.zmq_context.socket(
            zmq.DEALER)  # pylint: disable=no-member

        self.processing_queue: deque[ProcessingTaskArgs] = deque()

        # Tracks the currently running task
        self.current_task: AsyncTask = None
        self.current_uuid: str = None
        # Tracks if a compute engine is running
        self.compute_task = None
        self.processing_states: dict[str, ProcessingState] = {}
        self.processing_positions: dict[str, int] = {}
        self.task_timeout: dict[str, float] = {}

    def get_position(self, task_uuid: TaskUUID):
        """
        Fetch the position of a task in the processing queue

        *Useful for passing to callbacks that need to find out a tasks position
        in the `processing_queue`

        Args:
            task_uuid (TaskUUID): uuid associated with a processing task

        Raise:
            KeyError: When the task_uuid passed in is not in the
                processing_queue

        Returns:
            (int): position of task in the processing queue
        """

        return self.processing_positions[task_uuid]

    def get_queue_length(self):
        """
        Fetch the length of the current `processing_queue`

        *Useful for passing to callbacks that need access to the queue length

        Returns:
            (int): the length of the `processing_queue`
        """
        return len(self.processing_queue)

    def get_message_timeout(self, task_uuid: str):
        """
        Fetch a rough approximation of the number of seconds that have elapsed
            since the task associated `task_uuid` was added to the event loop

        *Useful for passing to callbacks that need to find out how long a
            task has until it times out

        Args:
            task_uuid (TaskUUID): uuid associated with a processing task

        Raise:
            KeyError: When the task_uuid passed in is not in the
                processing_queue
        Returns:
            (float): an approximation of the number of seconds since the task
                associated with `task_uuid` was added to the
                queue and event loop
        """

        return time.time() - self.task_timeout[task_uuid]

    def get_queue_timeout(self):
        """
        Get the current `queue_timeout` configuration

        *Useful for passing to callbacks that need access to the `queue_timeout`

        Returns:
            (int): the `queue_timeout` for the AsyncProcessingClient
        """
        return self.queue_timeout

    async def compute_engine(self):
        """
        A function meant to be ran in the background that will sequentially
        consume all processing tasks in the queue
        """

        # Execute all the remote image processing tasks in the processing_queue
        while self.processing_queue:
            processing_task_args = self.processing_queue.popleft()
            self.current_uuid = processing_task_args.task_uuid
            self.current_task = processing_task_args.compute_task

            if (self.processing_states.get(processing_task_args.task_uuid,
                                           None)
                    is ProcessingState.failed):
                del self.processing_states[processing_task_args.task_uuid]
                print(f"Deleting failed task {processing_task_args.task_uuid}")
            else:
                # if processing_task_args.callback_wrapper is not None:
                #     callback = processing_task_args.callback_wrapper.callback
                #     callback_args = processing_task_args.callback_wrapper.args
                #     await callback(*callback_args)

                self.processing_states[processing_task_args.task_uuid] = (
                    ProcessingState.in_progress)
                try:
                    # await processing_task_args.compute_task
                    await asyncio.wait([processing_task_args.compute_task],
                                       timeout=self.queue_timeout)
                # Ignore any exceptions as the compute_engine is running in
                #   its own thread and the exception will be raised in the main
                #   thread wherever the compute task was awaited
                except Exception as _exception:
                    print(f"Exception in compute engine {_exception}")
                # Remove processing state, task position and task timeout
                #   from caches now that task computation has completed
                finally:
                    _task_uuid = processing_task_args.task_uuid
                    if _task_uuid in self.processing_states:
                        del self.processing_states[processing_task_args.task_uuid]
                    if _task_uuid in self.processing_positions:
                        del self.processing_positions[
                            processing_task_args.task_uuid]
                    if _task_uuid in self.task_timeout:
                        del self.task_timeout[processing_task_args.task_uuid]

                    self.current_task = None
                    self.current_uuid = None
                await self.process_callbacks()

            print(f"Compute finished for: {processing_task_args.task_uuid},"
                  f" waiting on {len(self.processing_queue)} more tasks, "
                  f"{len(self.processing_states)} have state")

        self.compute_task = None
        # Wipe caches to avoid memory leaks
        self.processing_states = {}
        self.processing_positions = {}
        self.task_timeout = {}

    async def async_compute(self,
                            processing_args: ProcessingArgs,
                            address: str,
                            timeout: int,
                            callback_wrapper: CallbackWrapper | None,
                            task_uuid: TaskUUID | None = None,
                            index: int = APPEND_INDEX):
        """
        Add a processing task to the queue and wait until it returns or
        `queue_timeout` is reached

        Args:
            processing_args (ProcessingArgs): the arguments passed to the
                remote server
            address (str): the address the remote server is running at
            timeout (int): the time to wait after a connection has been
                initiated with the remote server, not to get confused with
                `queue_timeout`
            index (int, optional): The index to insert the processing task
                into the queue at. Defaults to APPEND_INDEX.

        Returns:
            (ProcessingResponse): a response from the remote server

        Raises:
            TimeoutError: raised when `queue_timeout` has been exceeded
        """
        if task_uuid is None:
            task_uuid = str(uuid4())

        image_processing_task = self._async_remote_compute(
            address, processing_args, timeout, task_uuid)

        processing_task_args = ProcessingTaskArgs(
            image_processing_task, task_uuid, callback_wrapper)
        if index == PRIORITY_INDEX:
            self.processing_queue.appendleft(0, processing_task_args)
        elif index == APPEND_INDEX:
            self.processing_queue.append(processing_task_args)
        else:
            self.processing_queue.insert(index, processing_task_args)

        # If there is no currently running process restart the async engine
        if self.compute_task is None:
            async_compute_engine = self.compute_engine()
            self.compute_task = asyncio.create_task(async_compute_engine)

        # Wait for currently compute_task to finish
        try:
            self.task_timeout[task_uuid] = time.time()
            self.processing_states[task_uuid] = ProcessingState.not_started
            response = await asyncio.wait_for(image_processing_task,
                                              self.queue_timeout)
        except Exception as exception:
            self.processing_states[task_uuid] = ProcessingState.failed
            raise exception
        return response

    async def _async_remote_compute(self, address: str,
                                    processing_args: ProcessingArgs,
                                    timeout: int,
                                    task_uuid: TaskUUID):
        """
        A helper function that will allow a compute task to be started but
        block the task from beginning remote computation until the task has
        been reached in the queue

        Args:
            address (str): address to connect to remote process on
            processing_args (ProcessingArgs): args to send to remote process
            timeout (int): timeout to wait for remote process
            task_uuid (TaskUUID): unique id associated with this task, that is
                track the currently running task in the compute_engine

        Returns:
            (ProcessingResponse): response from remote server

        Raises:
            Exception: raised when remote task takes longer than `timeout`
        """
        # If it is not time for this task to compute block until it has
        #   been "scheduled"
        while self.processing_states[task_uuid] != ProcessingState.in_progress:

            # If current_uuid is None then a task is not running to wait on
            if (self.current_uuid is not None):

                # if current_uuid == task_uuid then throw exception,
                #   processing queue has reached an impossible state, when
                #   current_uuid == task_uuid ProcessingState should be
                #   in_progress
                if (self.current_uuid == task_uuid):
                    raise RuntimeError(
                        f"State was not changed to "
                        f"{ProcessingState.in_progress} for currently running "
                        f"task {task_uuid}. Compute Engine must change a tasks "
                        "ProcessState to 'in_progress' before awaiting the "
                        "task")
                # await currently running task
                self.current_task
                done, _pending = await asyncio.wait([self.current_task],
                                                    timeout=self.queue_timeout)
                for task_set in done:
                    task_exception = task_set.exception()
                    if (task_exception is not None and
                        "cannot reuse already awaited coroutine" not in
                            str(task_exception)):
                        print(
                            (f"Exception while {task_uuid} is waiting "
                             f"on ({self.current_uuid})"
                             f"{self.current_task}"))
                        task_set.print_stack()
                if len(done) > 0:
                    await asyncio.sleep(0.1)

            else:
                await asyncio.sleep(0.1)

        self.zmq_socket.connect(address)
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, timeout)

        self.zmq_socket.send_string(json.dumps(processing_args))
        received = await self.zmq_socket.recv()
        processing_response = ProcessingResponse.from_bytes(received)

        return processing_response

    async def process_callbacks(self):
        """
        Process all callbacks in the processing_queue using a threadpool
        """

        start_time = time.time()
        callback_count = 0

        # with concurrent.futures.ThreadPoolExecutor() as executor:
        # futures_list: list[asyncio.Future] = []
        callback_list: list[Coroutine] = []
        for queue_index, task_args in enumerate(self.processing_queue):
            self.processing_positions[task_args.task_uuid] = queue_index
            if task_args.callback_wrapper is not None:
                callback_count += 1
                callback_task = task_args.callback_wrapper.callback(
                    task_args.callback_wrapper.args)
                callback_list.append(callback_task)
                # callback_future = executor.submit(
                #     task_args.callback_wrapper.callback,
                #     task_args.callback_wrapper.args)
                # futures_list.append(callback_future)
            # done_task, pending_task = concurrent.futures.wait(futures_list)
            # print("Coroutine tasks")
            # for task in done_task:
            #     print(task.exception())

        if len(callback_list) > 0:
            await asyncio.gather(*callback_list)

        print((f"Processed {callback_count} callbacks for "
               f"{len(self.processing_queue)} "
               f"tasks in {time.time() - start_time: .2f}s"))
