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

if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REQ)  # pylint: disable=no-member

    # if check_socket(GV.ZMQ_HOST, GV.ZMQ_PORT):
    socket.connect("tcp://localhost:5555")
    socket.send_string(" ".join(sys.argv[1:]))

    json_dict = json.loads(socket.recv_json())
    # else:
    # GV.global_parse_args(" ".join(sys.argv[1:]))
    # json_dict = detect.detect_features(GV.IMAGE_SS,
    #                                     detect_faction=False)

    print(json_dict)