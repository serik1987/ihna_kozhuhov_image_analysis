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


class ProgressBar:

    def progress_function(self, processed, total, message):
        print("{0} / {1}: {2}".format(processed, total, message))
        return True

    def __del__(self):
        print("PROGRESS BAR DESTRUCTION")


if __name__ == "__main__":

    animal_list = manifest.Animals("/home/serik1987/vasomotor-oscillations")
    animal = animal_list['c022']
    case_list = manifest.CasesList(animal)
    case = case_list['02']
    filename = os.path.join(case['pathname'], case['native_data_files'][0])
    roi = case['roi']['tissue']
    print(roi)

    train = files.StreamFileTrain(filename)
    train.open()
    sync = synchr.ExternalSynchronization(train)
    sync.channel_number = 1
    isoline = isolines.TimeAverageIsoline(train, sync)
    accumulator = acc.TraceAutoReader(isoline)
    bar = ProgressBar()
    accumulator.set_roi(roi)
    accumulator.set_progress_bar(bar)

    print(accumulator)
    print("PY Channel number: ", accumulator.channel_number)
    print("PY Accumulated: ", accumulator.is_accumulated)
    accumulator.accumulate()
    try:
        print(accumulator.averaged_signal)
    except Exception as err:
        print(err)

    del bar
    print("PY Progress bar destroyed")

    del train
    print("PY Train destroyed")

    del sync
    print("PY Synchronization destroyed")

    del isoline
    print("PY Isoline destroyed")

    del accumulator
    print("PY Accumulator destroyed")

    print("PY Test end")
