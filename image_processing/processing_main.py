import sys

from image_processing.processing.processing_client import ProcessingClient
from image_processing.processing.processing_server import ProcessingServer


if __name__ == "__main__":
    processing_result = ProcessingServer.compute(sys.argv[1:])
    ProcessingClient.print_result(processing_result)
