"""
Helper Script that can be used to connect to the feature detection server when
it is online, or start a new temporary feature detection section to parse the
image/arguments passed to this script
"""
import sys
import json
import pprint
from typing import List

import zmq
import jsonpickle

from image_processing.afk.hero.hero_data import RosterJson
import image_processing.globals as GV


def remote_compute_results(address: str, timeout: int, args: List[str]):
    """_summary_

    Args:
        address (str): address to connect the socket to
        timeout (int): timeout in ms to wait for results
        args (str): local path to image, or discord image URL and other global
            args
    """

    zmq_context = zmq.Context()
    zmq_socket: zmq.Socket = zmq_context.socket(
        zmq.DEALER)  # pylint: disable=no-member

    zmq_socket.connect(address)
    # pylint: disable=no-member
    zmq_socket.setsockopt(zmq.RCVTIMEO, timeout)

    print(f"Arguments: {args}")

    zmq_socket.send_string(json.dumps(args))
    received = zmq_socket.recv()
    roster_json_str = received.decode("utf-8")
    return roster_json_str


def main():
    """_summary_
    """

    address = "tcp://localhost:5555"
    GV.global_parse_args()

    roster_json_str = remote_compute_results(address, 15000, sys.argv[1:])

    try:
        roster_json = RosterJson.from_json(roster_json_str)
        pprint.pprint(roster_json.json_dict(), width=200)

    # pylint: disable=broad-except
    except Exception as _serialization_exception:
        error_message = jsonpickle.decode(roster_json_str)
        print(error_message["message"])


if __name__ == "__main__":
    main()
