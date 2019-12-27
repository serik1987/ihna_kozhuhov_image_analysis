#!/usr/bin/env python3

import ihna.kozhukhov.imageanalysis.sourcefiles as files

if __name__ == "__main__":
    print("PY Test begin")

    train = files.CompressedFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0A00z", "traverse")
    train.open()

    for file in train:
        print(file)
        for chunk in file.isoi:
            print(chunk)

    print("PY Test end")
