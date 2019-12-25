#!/usr/bin/env python3

import ihna.kozhukhov.imageanalysis.sourcefiles as files

if __name__ == "__main__":
    print("PY Test begin")

    train = files.CompressedFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0A00z", "traverse")
    train.open()
    print(train)

    for file in train:
        print(file)
        print(file.isoi)

    comp = file.isoi['comp']
    print("PY comp chunk extracted")
    print(comp)
    print("PY Chunk name: ", comp['id'])
    print("PY Chunk size: ", comp['size'])
    print("PY Single extrapixel size: ", comp['compressed_record_size'])
    print("PY Compressed frame size: ", comp['compressed_frame_size'])
    print("PY Compressed frame number: ", comp['compressed_frame_number'])

    print("PY Test end")
