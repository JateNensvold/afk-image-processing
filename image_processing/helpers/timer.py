import time
import collections

from typing import Dict, List, NamedTuple


class TimerTuple(NamedTuple):

    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """
    info: str
    start: int
    end: int

    def __str__(self):
        """_summary_

        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            _type_: _description_
        """
        return f"{self.info}, {self.start}->{self.end}:{self.end-self.start}"

    def time_length(self):
        """_summary_

        Raises:
            Exception: _description_
            Exception: _description_

        Returns:
            _type_: _description_
        """
        return self.end - self.start


class TimerEvent(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """
    level: int
    name: str


class TimeDict:
    """_summary_
    """

    def __str__(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        str_list = []
        str_list.append(str(self.event_info))
        indent = self.event_info.level
        tab = "\t"
        for time_event in self.times:
            str_list.append(str(time_event))
        str_list.append(f"Total: {self.total_time}")
        for str_index, str_item in enumerate(str_list):
            str_list[str_index] = f"{tab*indent}{str_item}"
        return "\n".join(str_list)

    def __init__(self, event_info: TimerEvent):
        """_summary_

        Args:
            event_info (TimerEvent): _description_
        """
        self.event_info = event_info

        self.times: List[TimerTuple] = []

        # Tracks epoch time
        self.end_time: int = None
        self.start_time: int = time.time()
        self._total_time = 0
        # Tracks current event
        self.current_info: List[str] = None
        # Holds start time of events in progress
        self.current_start: int = None

    @property
    def total_time(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        if self.end_time is None:
            return self._total_time
        else:
            return self.end_time - self.start_time

    def start(self, info: str = None):
        """_summary_

        Args:
            info (str, optional): _description_. Defaults to None.
        """
        if self.current_info is not None and self.current_start is not None:
            raise Exception(
                "Timer event already in progress, call `stop` before calling "
                "start again")
        self.current_info = []
        if info:
            self.current_info.append(info)
        self.current_start = time.time()

    def stop(self, info: str = None):
        """_summary_
        """
        if self.current_info is None and self.current_start is None:
            raise Exception(
                "Timer event has not been started, call `start` before calling"
                " `stop`")
        stop_time = time.time()
        if info:
            self.current_info.append(info)

        if len(self.current_info) == 0:
            self.current_info.append(self.event_info.name)

        timer_tuple = TimerTuple(
            ". ".join(self.current_info), self.current_start, stop_time)

        self.times.append(timer_tuple)
        self._total_time += timer_tuple.time_length()
        self.current_info = None
        self.current_start = None

    def end(self, info: str = None):
        """_summary_
        """

        if self.current_start is None and self.current_info is None:
            # Stop any ongoing timers if end is called
            self.stop(info)

        self.end_time = time.time()


class Timer:
    """_summary_
    """

    def __init__(self, name: str = "main"):
        """_summary_
        """
        start_level = TimeDict(TimerEvent(0, name))
        self.current_level = start_level
        self.start_level = start_level

        self.time_stack: collections.defaultdict[
            int, List[TimeDict]] = collections.defaultdict(list)
        self.time_stack[start_level.event_info.level].append(start_level)
        self.start_level.start()

    def start(self, info: str = None):
        """_summary_
        """
        self.current_level.start(info)

    def stop(self, info: str = None):
        """_summary_
        """
        self.current_level.stop(info)

    def add_level(self, level_name: str):
        """_summary_
        """
        self.current_level = TimeDict(TimerEvent(
            self.current_level.event_info.level + 1, level_name))
        self.time_stack[self.current_level.event_info.level].append(
            self.current_level)

    def finish_level(self, info: str = None):
        """_summary_
        """
        self.current_level.end(info)
        if self.current_level.event_info.level > 0:
            self.current_level = self.time_stack[
                self.current_level.event_info.level - 1][-1]

    def display(self):
        """_summary_
        """
        current_level = self.start_level.event_info.level

        while current_level in self.time_stack:
            for events in self.time_stack[current_level]:
                print(str(events))
            current_level += 1

    def display_breakdown(self):
        """_summary_
        """

        breakdown_dict: Dict[TimerEvent, int] = collections.defaultdict(int)

        current_level = self.start_level.event_info.level

        while current_level in self.time_stack:
            for events in self.time_stack[current_level]:
                breakdown_dict[events.event_info] += events.total_time
            current_level += 1

        breakdown_str_list = []
        total_time = 0
        tab = "\t"
        for event_info, event_time in breakdown_dict.items():
            breakdown_str_list.append((
                event_info.level,
                f"{tab*event_info.level}{event_info.name}: {event_time}"))
            total_time += event_time
        breakdown_str_list.sort(key=lambda x: x[0])

        print("\n".join([str_object for level,
              str_object in breakdown_str_list]))
