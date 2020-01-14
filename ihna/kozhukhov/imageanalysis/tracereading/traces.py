# -*- coding: utf-8


class Traces:
    """
    Copies traces from the Trace Reader and provides their processing. The resultant signal contains
    only temporary-dependent data from the synchronization channel and from the pixels given at a frame

    Construction:
    Traces(case) - load the traces from the hard disk after they have been processed
    Traces(case, processor) - get the traces from the processor
    """

    def __init__(self, case, processor=None):
        print("Trace analysis")
