from enum import Enum


class ProcessingStatus(Enum):
    success: int = 0
    failure: int = 1
    reload: int = 2
