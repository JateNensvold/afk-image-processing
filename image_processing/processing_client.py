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
    GV.global_parse_args()
    if run_detection:
        return detect.detect_features(GV.IMAGE_SS,
                                      detect_faction=False)


if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # pylint: disable=no-member

    detect_locally = True
    if check_socket(GV.ZMQ_HOST, GV.ZMQ_PORT):
        detect_locally = False

    json_dict = process_image(detect_locally)

    if not detect_locally:
        socket.connect("tcp://localhost:5555")
        socket.send_string(" ".join(sys.argv[1:]))

        json_dict = json.loads(socket.recv().decode("utf-8"))
    print(json_dict)
