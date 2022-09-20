"""
Script used to start the feature detection server

This script provides a way for outside clients or even local clients to connect
to an environment that is already initialized and also has the Image database
and ML models already loaded
"""
import json
import time
import traceback
from typing import List

import zmq

import image_processing.globals as GV
import image_processing.utils.load_models as LM
import image_processing.afk.detect_image_attributes as detect
from image_processing.build_db import load_database
from image_processing.processing.async_processing.processing_status import (
    ProcessingStatus)
from image_processing.processing.async_processing.processing_response import (
    ProcessingResponse)

DATABASE_LOAD_MESSAGE = "Database loaded successfully"
RELOAD_COMMAND_LIST = ["reload"]


class ProcessingServer:
    def __init__(self, host: str | None = None, port: int = GV.ZMQ_PORT):
        """
        Initialize all the ZMQ variables to allow for the image processing
        server to run

        Args:
            host (str | None): address to listen on, when None will listen
                on all address. Defaults to None. Host cannot be `localhost`
                when binding, using ipv4 equivalent instead
            port (int): port to listen on. Defaults to GV.ZMQ_PORT
        """
        self.context = zmq.Context()
        # Rep Socket can only receive a single message at a time
        self.socket: zmq.Socket = self.context.socket(zmq.ROUTER)
        if host is None:
            host = "*"
        self.host = host
        self.port = port
        self.address = f"tcp://{self.host}:{self.port}"
        print(f"Binding on ({self.address})")

        self.socket.bind(self.address)

    def listen(self):
        """
        Setup the processing server to listen for requests indefinitely
        """

        GV.VERBOSE_LEVEL = 1
        LM.load_files(str(GV.FI_SI_STARS_MODEL_PATH),
                      str(GV.ASCENSION_BORDER_MODEL_PATH))

        print("Ready to start listening to image requests...")
        try:
            while True:
                self.run()
        except Exception as _exception:
            exception_message = traceback.format_exc()
            print(f"Aborting processing server due to \n\n{exception_message}")

    def run(self):
        """
        Listen for and process a single request
        """

        #  Wait for next request from client
        # pylint: disable=unpacking-non-sequence
        # output = self.socket.recv_multipart()
        # print(output)
        # # job_id
        # message_id = output[0]
        # byte_args = output
        output = self.socket.recv_multipart()
        print(output)
        # Dealer response
        if len(output) == 2:
            message_id, byte_args = output
            message_code = "No code"

        else:
            message_id, message_code, byte_args = output
        print(f"Received message: {message_id} with code ({message_code})")
        args: List[str] = json.loads(byte_args)

        response = self.compute(args)
        self.socket.send_multipart([message_id, response.to_bytes()])

    @classmethod
    def compute(cls, args: list[str]):
        """
        Run image_processing on list of arguments

        Args:
            args (list[str]): list of arguments to process on

        Returns:
            str: the response from running image processing
        """

        if args == RELOAD_COMMAND_LIST:
            database_reload = True
        else:
            database_reload = False
        try:
            if database_reload:
                GV.IMAGE_DB = load_database()
                return ProcessingResponse(ProcessingStatus.reload,
                                          result=None,
                                          message=DATABASE_LOAD_MESSAGE)
            else:
                GV.global_parse_args(args)
                start_time = time.time()
                roster_data = detect.detect_features(GV.IMAGE_SS)
                detection_message = (
                    f"Detected features in: {time.time() - start_time}")
                print(detection_message)
                return ProcessingResponse(ProcessingStatus.success,
                                          result=roster_data,
                                          message=detection_message)

        # Catch all errors that occur during image processing and build
        #    appropriate response
        except Exception as _exception:  # pylint: disable=broad-except
            exception_message = traceback.format_exc()
            print(exception_message)
            return ProcessingResponse(ProcessingStatus.failure,
                                      result=None,
                                      message=exception_message)


if __name__ == "__main__":
    processing_server = ProcessingServer(GV.ZMQ_HOST, GV.ZMQ_PORT)
    processing_server.listen()
