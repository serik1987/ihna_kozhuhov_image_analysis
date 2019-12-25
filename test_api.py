#!/usr/bin/env python3

import ihna.kozhukhov.imageanalysis.sourcefiles as files

if __name__ == "__main__":
    print("PY Test begin")

    train = files.CompressedFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0A00z", "traverse")
    train.open()

    for file in train:
        print(file.filename)

    chunk = file.isoi['cost']
    print("PY cost chunk has been created")
    print("PY Chunk identifier: ", chunk['id'])
    print("PY Chunk size: ", chunk['size'])
    print("PY Synchronization channel number: ", chunk['synchronization_channel_number'])
    print("PY Synchronization channels max: ", chunk['synchronization_channel_max'])
    print("PY Total number of stimulus channels: ", chunk['stimulus_channels'])
    print("PY Stimulus periods: ", chunk['stimulus_period'])

    print("PY Test end")
