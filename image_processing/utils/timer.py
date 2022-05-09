import time

from typing import NamedTuple


class TimerCache:

    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """

    def __init__(self, parent_cache: "TimerCache"):

        self.parent_cache = parent_cache
        self.time = 0
        self.cache: dict[str, "TimerCache"] = {}

    def __contains__(self, key: str):
        """_summary_

        Args:
            key (str): _description_

        Returns:
            _type_: _description_
        """
        if key in self.cache:
            return True
        return False

    def __setitem__(self, name: str, value: "TimerCache") -> None:
        """_summary_

        Args:
            name (str): _description_
            value (TimerCache): _description_
        """
        self.cache[name] = value

    def __getitem__(self, name: str) -> "TimerCache":

        try:
            return self.cache[name]
        except KeyError as exception:
            raise KeyError(f"{name} not found in TimerCache") from exception

    def display(self, name: str, indent: int):
        """_summary_

        Args:
            indent (int): _description_
        """

        tab_char = "\t"

        print(f"{tab_char*indent}{name}:{round(self.time, 4)}")

        for cache_name, cache in self.cache.items():
            cache.display(cache_name, indent + 1)

    @property
    def len(self):
        """_summary_

        Returns:
            _type_: _description_
        """

        return self.time


class TimerEvent(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """
    recursion_level: int
    time_dict: "TimeDict"


class TimeDict:
    """_summary_
    """

    def __str__(self):
        """_summary_
        """

        return f"[D] {self.event_name}: {self.len}"

    def __init__(self, event_name: str):
        """_summary_

        Args:
            event_name (str): _description_
        """
        self.event_name = event_name

        self.start_time: int = None
        self.stop_time: int = None

        self._total_time = 0

    @property
    def len(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # Return best approximate of elapsed time while class is in flight
        if self.stop_time is None:
            return time.time() - self.start_time
        else:
            return self._total_time

    def start(self):
        """_summary_

        Args:
            info (str, optional): _description_. Defaults to None.
        """
        if self.start_time:
            raise Exception(
                "Timer event already in progress, call `stop` before calling "
                "start again")

        self.start_time = time.time()

    def stop(self):
        """
        Stop current timer
        """
        stop_time = time.time()

        if self.start_time is None:
            raise Exception(
                "Timer event has not been started, call `start` before calling"
                " `stop`")

        self.stop_time = stop_time
        self._total_time += self.stop_time - self.start_time


class Timer:
    """_summary_
    """

    def __init__(self):
        """_summary_
        """
        self.current_timer: TimeDict = None
        self.current_cache = TimerCache(None)
        self.timer_stack = []

    def start(self, info: str, reset: bool = False):
        """
        Start a new timer
        """

        if reset:
            self.current_cache = TimerCache(None)
            self.current_timer = None

        if self.current_timer:
            self.timer_stack.append(self.current_timer)

        if info not in self.current_cache:
            new_cache = TimerCache(self.current_cache)
            self.current_cache[info] = new_cache

        self.current_cache = self.current_cache[info]

        self.current_timer = TimeDict(info)

        self.current_timer.start()

    def stop(self):
        """
        Stop current timer
        """
        self.current_timer.stop()
        self.current_cache.time += self.current_timer.len

        if self.timer_stack:
            self.current_timer = self.timer_stack.pop()
            self.current_cache = self.current_cache.parent_cache
        else:
            self.current_timer = None

    def display(self):
        """_summary_
        """

        self.current_cache.display(self.current_timer.event_name, 0)
