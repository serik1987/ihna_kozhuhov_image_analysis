#!/usr/bin/env python3

import ihna.kozhukhov.imageanalysis as iman
import ihna.kozhukhov.imageanalysis.sourcefiles as files

'''
try:
    iman.test_exception()
except iman.ImanError as e:
    print("Error message: ", e)
    print("Error class name: ", e.__class__.__name__)
    print("Error class: ", e.__class__)
    if isinstance(e, files.IoError):
        print("Instance of files.IoError")
    if isinstance(e, files.TrainError):
        print("Instance of files.TrainError")
        print("Train name: ", e.train_name)
    if isinstance(e, files.SourceFileError):
        print("Instance of files.SourceFileError")
        print("File name: ", e.file_name)
    if isinstance(e, files.ChunkError):
        print("Instance of files.ChunkError")
        print("Chunk name: ", e.chunk_name)
'''

if __name__ == "__main__":
    print("PY Test begin")

    train = files.CompressedFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0A00z", "traverse")
    train.open()
    print(train)

    for file in train:
        print(file.filename)

    isoi = file.isoi
    print(isoi)

    soft = isoi['FRAM']
    print(soft)

    print("PY Test end")
