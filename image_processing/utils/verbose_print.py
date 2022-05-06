import image_processing.globals as GV


def print_verbose(message: str, verbose_level: int = 1):
    """
    Prints a message when the verbose level of the program is above the
        'verbose_level' passed into this function
    Args:
        message: message to print
        verbose_level: minimum level of verbosity to print message at
    """

    if GV.verbosity(verbose_level):
        print(message)
