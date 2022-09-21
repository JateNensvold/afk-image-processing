import json
from typing import NamedTuple

import jsonpickle

from image_processing.afk.hero.hero_data import RosterJson
from image_processing.processing.async_processing.processing_status import (
    ProcessingStatus)


class ProcessingResponse(NamedTuple):
    """
    A class wrapping the results of an image processing computation 
    """

    status: ProcessingStatus
    result: RosterJson | None
    message: str

    def to_dict(self):
        """
        Convert the ProcessingResponse into a serializable dictionary
        """
        response_dict: dict[str, str | int] = {}

        response_dict["status"] = self.status.value
        if self.status == ProcessingStatus.success:
            response_dict["result"] = self.result.json()
        else:
            response_dict["result"] = None
        response_dict["message"] = jsonpickle.encode(self.message)
        return response_dict

    def to_str(self):
        """
        Convert the ProcessingResponse into a str
        """
        response_dict = self.to_dict()
        return json.dumps(response_dict)

    def to_bytes(self):
        """
        Convert the ProcessingResponse into a byte-stream
        """
        return self.to_str().encode("utf-8")

    @classmethod
    def from_bytes(cls, response: bytes):
        """
        Create a ProcessingResponse from bytes

        Args:
            response (bytes): bytes to turn into ProcessingResponse

        Returns:
            ProcessingResponse: new ProcessingResponse object
        """

        return cls.from_str(response.decode("utf-8"))

    @classmethod
    def from_str(cls, response: str):
        """
        Create a ProcessingResponse from a serialize dictionary stored as string
        """

        response_dict = json.loads(response)
        return cls.from_dict(response_dict)

    @classmethod
    def from_dict(cls, response: dict):
        """
        Create a ProcessingResponse from a dictionary
        """

        raw_status = int(response["status"])
        raw_result = response["result"]
        raw_message = response["message"]

        status = ProcessingStatus(raw_status)
        if status == ProcessingStatus.success:
            result = RosterJson.from_json(raw_result)
        else:
            result = raw_result
        message = jsonpickle.decode(raw_message)

        return ProcessingResponse(status, result, message)
