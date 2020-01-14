# -*- coding: utf-8


class TraceProcessor:
    """
    Provides an interface for post-processing of traces after they have been read
    The trace postprocessing is based on SCIPY
    """

    def __init__(self, reader):
        print("")