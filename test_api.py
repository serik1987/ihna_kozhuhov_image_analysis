#!/usr/bin/env python3

import matplotlib.pyplot as plt
import ihna.kozhukhov.imageanalysis as iman
import ihna.kozhukhov.imageanalysis.sourcefiles as files
import ihna.kozhukhov.imageanalysis.compression as comp
import ihna.kozhukhov.imageanalysis.tracereading as trace
from ihna.kozhukhov.imageanalysis import manifest
from ihna.kozhukhov.imageanalysis import synchronization as sync
from ihna.kozhukhov.imageanalysis.gui import synchronization as sync_edt


def progress_bar(perc):
    print("{0} percent completed".format(perc))


if __name__ == "__main__":
    print("PY Test begin")

    train = files.StreamFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0200")
    train.open()

    trace_reader = trace.TraceReader(train)



    print("PY Test end")
