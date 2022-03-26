"""
Script used to start the feature detection server

This script provides a way for outside clients or even local clients to connect
to an environment that is already initialized and also has the Image database
and already loaded
"""
import json
import time
import traceback
from typing import List

import zmq

import image_processing.globals as GV
import image_processing.helpers.load_models as LM
import image_processing.afk.detect_image_attributes as detect


def main():
    """_summary_
    """
    context = zmq.Context()
    socket = context.socket(zmq.ROUTER)  # pylint: disable=no-member
    address = f"tcp://*:{GV.ZMQ_PORT}"
    print(f"Binding on ({address})")

    socket.bind(address)

    GV.VERBOSE_LEVEL = 1
    LM.load_files(str(GV.FI_SI_STARS_MODEL_PATH),
                  str(GV.ASCENSION_BORDER_MODEL_PATH))

    print("Ready to start processing image requests...")
    while True:
        #  Wait for next request from client
        # pylint: disable=unbalanced-tuple-unpacking
        message_id, byte_args = socket.recv_multipart()
        args: List[str] = json.loads(byte_args)
        try:
            GV.global_parse_args(args)
            start_time = time.time()
            roster_data = detect.detect_features(GV.IMAGE_SS)

            print(f"Detected features in: {time.time() - start_time}")
            #  Send reply back to client
            socket.send_multipart(
                [message_id, json.dumps(roster_data.json()).encode("utf-8")])
        except Exception as _exception:
            exception_message = traceback.format_exc()
            print(exception_message)
            json_dict = {"message": exception_message}
            socket.send_multipart(
                [message_id, json.dumps(json_dict).encode("utf-8")])


if __name__ == "__main__":
    main()
