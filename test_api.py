#!/usr/bin/env python3

import os
import matplotlib.pyplot as plt
import ihna.kozhukhov.imageanalysis as iman
import ihna.kozhukhov.imageanalysis.sourcefiles as files
import ihna.kozhukhov.imageanalysis.compression as comp
import ihna.kozhukhov.imageanalysis.tracereading as trace
from ihna.kozhukhov.imageanalysis import manifest
from ihna.kozhukhov.imageanalysis import synchronization as synchr
from ihna.kozhukhov.imageanalysis import isolines
import ihna.kozhukhov.imageanalysis.accumulators as acc


def progress_bar(completed, total, message):
    print("{0}: {1} percent completed".format(message, completed/total))


if __name__ == "__main__":

    train = files.StreamFileTrain("/home/serik1987/vasomotor-oscillations/c022/T_1BF.0200")
    train.open()
    sync = synchr.ExternalSynchronization(train)
    sync.channel_number = 1
    isoline = isolines.TimeAverageIsoline(train, sync)
    accumulator = acc.TraceAutoReader(isoline)

    print(accumulator)

    del train
    print("PY Train destroyed")

    del sync
    print("PY Synchronization destroyed")

    del isoline
    print("PY Isoline destroyed")

    del accumulator
    print("PY Accumulator destroyed")

    print("PY Test end")
