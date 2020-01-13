#!/usr/bin/env python3

import matplotlib.pyplot as plt
import ihna.kozhukhov.imageanalysis as iman
import ihna.kozhukhov.imageanalysis.sourcefiles as files
import ihna.kozhukhov.imageanalysis.compression as comp
import ihna.kozhukhov.imageanalysis.tracereading as trace
from ihna.kozhukhov.imageanalysis import manifest
from ihna.kozhukhov.imageanalysis import synchronization as synchr
from ihna.kozhukhov.imageanalysis import isolines
from ihna.kozhukhov.imageanalysis.gui import isolines as isoline_editors
from ihna.kozhukhov.imageanalysis.gui.isolines.selector import IsolineSelector


def progress_bar(completed, total, message):
    print("{0}: {1} percent completed".format(message, completed/total))


if __name__ == "__main__":
    print("PY Test begin")

    train = files.StreamFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0200")
    train.open()

    trace_reader = trace.TraceReaderAndCleaner(train)
    print(trace_reader)

    del train
    print("PY Train was deleted")

    del trace_reader
    print("PY Trace reader was deleted")

    print("PY Test end")
