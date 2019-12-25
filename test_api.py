#!/usr/bin/env python3

import ihna.kozhukhov.imageanalysis.sourcefiles as files

if __name__ == "__main__":
    print("PY Test begin")

    train = files.CompressedFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0A00z", "traverse")
    train.open()

    for file in train:
        print(file.filename)

    chunk = file.isoi['data']
    print(chunk)
    print("PY chunk has been created")
    print("PY Chunk ID: ", chunk["id"])
    print("PY Chunk size: ", chunk['size'])

    del train
    print("PY train has been deleted")

    del file
    print("PY file has been deleted")

    del chunk
    print("PY Chunk has been deleted")

    print("PY Test end")
