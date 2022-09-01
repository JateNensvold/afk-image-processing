"""
Helper Script that can be used to connect to the feature detection server when
it is online, or start a new temporary feature detection section to parse the
image/arguments passed to this script
"""
import sys
import json
import pprint
from typing import List

from image_processing.processing.async_processing.processing_response import (
    ProcessingResponse)
from image_processing.processing.async_processing.processing_status import (
    ProcessingStatus)

import zmq

import image_processing.globals as GV


class ProcessingClient:

    def __init__(self):
        """
        Initialize all the ZMQ variables to allow for the processing
        client to run
        """
        self.zmq_context = zmq.Context()
        self.zmq_socket: zmq.Socket = self.zmq_context.socket(
            zmq.DEALER)  # pylint: disable=no-member

    def remote_compute_results(self, address: str,
                               timeout: int,
                               args: List[str]):
        """
        Connect to remote processing server and run image recognition

        Args:
            address (str): address to connect the socket to
            timeout (int): timeout in ms to wait for results
            args (str): local path to image, or discord image URL and other
                global args

        Returns:
            processing_response: response from remote computation
        """
        # zmq_context = zmq.Context()
        # zmq_socket: zmq.Socket = zmq_context.socket(
        #     zmq.DEALER)  # pylint: disable=no-member

        self.zmq_socket.connect(address)
        # pylint: disable=no-member
        self.zmq_socket.setsockopt(zmq.RCVTIMEO, timeout)

        print(f"Arguments: {args}")

        self.zmq_socket.send_string(json.dumps(args))
        received = self.zmq_socket.recv()
        processing_response = ProcessingResponse.from_bytes(received)
        return processing_response

    @classmethod
    def print_result(cls, processing_response: ProcessingResponse):
        """
        Print a ProcessingResponse

        Args:
            processing_response (ProcessingResponse): response to process 
                and print
        """

        if processing_response.status == ProcessingStatus.success:
            pprint.pprint(processing_response.result.json_dict(), width=200)
        else:
            print(processing_response.message)


if __name__ == "__main__":
    """
    Run the processing client with arguments from the CMD line
    """
    args = sys.argv[1:]
    timeout = 15000
    address = f"tcp://{GV.ZMQ_HOST}:{GV.ZMQ_PORT}"

    processing_client = ProcessingClient()
    processing_response = processing_client.remote_compute_results(
        address=address, timeout=timeout, args=args)

    processing_client.print_result(processing_response)
