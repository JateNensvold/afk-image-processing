"""
Helper Module used to check if a port is open at a remote address
"""
import time
import socket


def check_socket(address: str, port: int, timeout: int = 1):
    """
    Check if the 'port' located on 'ip' is open

    Args:
        address (str): ip to check port on
        port (int): port to check
        timeout (int, optional): Timeout(seconds) to scan port for before
            returning port unavailable. Defaults to 1.

    Returns:
        [bool]: True if port is open for connections, False otherwise
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = -1
    start_time = time.time()
    while result != 0 and (start_time + timeout) < time.time():
        result = sock.connect_ex(address, port)
    sock.close()

    return result == 0
