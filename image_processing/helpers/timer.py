import time
from typing import NamedTuple


class TimerTuple(NamedTuple):
    """_summary_

    Args:
        NamedTuple (_type_): _description_
    """
    level: int
    name: str


class Timer:
    """_summary_
    """

    def __init__(self, name: str = "main"):
        """_summary_
        """

        self.time_history = []
        self.time_dict = {}
        self.current_stack = []

        self.time_stack = []
        self.current_level = TimerTuple(0, name)
        self.add_level(self.current_level)

    def start(self):
        """_summary_
        """
        self.current_stack.append(time.time())

    def stop(self):
        """_summary_
        """
        self.current_stack.append(time.time())

    def add_level(self, current_level: TimerTuple):
        """_summary_
        """
        self.current_level = current_level
        self.time_stack.append(self.current_level)
        self.current_stack = []

        self.time_dict[current_level] = self.current_stack

    def finish_level(self):
        """_summary_
        """
        _last_level = self.time_stack.pop()
        self.current_stack = self.time_dict[self.time_stack[-1]]
