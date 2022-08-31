"""
Script used to start the feature detection server

This script provides a way for outside clients or even local clients to connect
to an environment that is already initialized and also has the Image database
and ML models already loaded
"""
import json
import time
import traceback
from typing import List
from image_processing.build_db import load_database

import zmq
import jsonpickle

import image_processing.globals as GV
import image_processing.utils.load_models as LM
import image_processing.afk.detect_image_attributes as detect


DATABASE_LOAD_MESSAGE = "Database loaded successfully"
RELOAD_COMMAND_LIST = ["reload"]

def main():
    """
    Setup the processing server to listen for requests
    """
    context = zmq.Context()
    socket: zmq.sugar.socket.Socket = context.socket(zmq.ROUTER)
    address = f"tcp://*:{GV.ZMQ_PORT}"
    print(f"Binding on ({address})")

    socket.bind(address)

    GV.VERBOSE_LEVEL = 1
    LM.load_files(str(GV.FI_SI_STARS_MODEL_PATH),
                  str(GV.ASCENSION_BORDER_MODEL_PATH))

    print("Ready to start processing image requests...")
    while True:
        #  Wait for next request from client
        # pylint: disable=unpacking-non-sequence
        message_id, byte_args = socket.recv_multipart()
        args: List[str] = json.loads(byte_args)
        if args == RELOAD_COMMAND_LIST:
            database_reload = True
        else:
            database_reload = False
        try:
            if database_reload:
                GV.IMAGE_DB = load_database()
                json_dict = {"message": DATABASE_LOAD_MESSAGE}
                socket.send_multipart(
                    [message_id, jsonpickle.encode(json_dict).encode("utf-8")])
            else:
                GV.global_parse_args(args)
                start_time = time.time()
                roster_data = detect.detect_features(GV.IMAGE_SS)
                print(f"Detected features in: {time.time() - start_time}")
                #  Send reply back to client
                socket.send_multipart(
                    [message_id, roster_data.json().encode("utf-8")])

        # Catch all errors that occur during image processing and send
        #   through json_dict
        except Exception as _exception:  # pylint: disable=broad-except
            exception_message = traceback.format_exc()
            print(exception_message)
            json_dict = {"message": exception_message}
            socket.send_multipart(
                [message_id, jsonpickle.encode(json_dict).encode("utf-8")])


if __name__ == "__main__":
    main()
