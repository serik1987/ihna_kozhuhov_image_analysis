# -*- coding: utf-8


class Traces:
    """
    Copies traces from the Trace Reader and provides their processing. The resultant signal contains
    only temporary-dependent data from the synchronization channel and from the pixels given at a frame
    """

    def __init__(self, case, reader):
        print("Trace analysis")
