"""
Helper Script that can be used to connect to the feature detection server when
it is online, or start a new temporary feature detection section to parse the
image/arguments passed to this script
"""
import sys
import json

import zmq

import image_processing.globals as GV
import image_processing.afk.detect_image_attributes as detect
from image_processing.helpers.scan_port import check_socket


def process_image(run_detection: bool = True):
    """_summary_

    Args:
        run_detection (bool, optional): _description_. Defaults to True.

    Returns:
        _type_: _description_
    """
    GV.global_parse_args()
    if run_detection:
        return detect.detect_features(GV.IMAGE_SS)


def main():
    """_summary_
    """

    context = zmq.Context()
    socket: zmq.Socket = context.socket(
        zmq.DEALER)  # pylint: disable=no-member

    detect_locally = True
    # if check_socket(GV.ZMQ_HOST, GV.ZMQ_PORT):
    #     detect_locally = False

    # json_dict = process_image(detect_locally)

    # if not detect_locally:
    socket.connect("tcp://localhost:5555")
    socket.setsockopt(zmq.RCVTIMEO, 15000)

    message = " ".join(sys.argv[1:])
    print(f"Message: {message}")

    socket.send_string(message, zmq.NOBLOCK)
    received = socket.recv()
    json_dict = json.loads(received)
    print(json_dict)

if __name__ == "__main__":
    main()
