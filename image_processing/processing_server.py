"""
Script used to start the feature detection server

This script provides a way for outside clients or even local clients to connect
to an environment that is already initialized and also has the Image database
and already loaded
"""
import json
import time
from numpy import byte

import zmq

import image_processing.globals as GV
import image_processing.helpers.load_models as LM
import image_processing.afk.detect_image_attributes as detect

GV.DISABLE_ARGPARSE = True

if __name__ == "__main__":
    context = zmq.Context()
    socket = context.socket(zmq.REP)  # pylint: disable=no-member
    socket.bind(f"tcp://*:{GV.ZMQ_PORT}")

    GV.VERBOSE_LEVEL = 1
    LM.load_files(str(GV.FI_SI_STARS_MODEL_PATH),
                  str(GV.ASCENSION_BORDER_MODEL_PATH))
    while True:
        #  Wait for next request from client
        byte_message: byte = socket.recv()
        message = byte_message.decode("utf-8")
        GV.global_parse_args(message)
        start_time = time.time()
        json_dict = detect.detect_features(GV.IMAGE_SS, detect_faction=False)
        print(f"Detected features in: {time.time() - start_time}")

        #  Send reply back to client
        socket.send(json.dumps(json_dict).encode("utf-8"))
        # socket.send_json(json.dumps(json_dict))
