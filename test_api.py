#!/usr/bin/env python3

import ihna.kozhukhov.imageanalysis.sourcefiles as files

if __name__ == "__main__":
    print("PY Test begin")

    train = files.CompressedFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0A00z", "traverse")
    train.open()

    for file in train:
        print(file.filename)

    chunk = files.EpstChunk()
    print(chunk)
    print("PY Chunk ID: ", chunk['id'])
    print("PY Chunk size: ", chunk['size'])
    print("PY Condition number: ", chunk['condition_number'])
    print("PY Repetition number: ", chunk['repetition_number'])
    print("PY Is randomized: ", chunk['randomized'])
    print("PY ITI frames: ", chunk['iti_frames'])
    print("PY Stimulus frames: ", chunk['stimulus_frames'])
    print("PY Prestimulus frames: ", chunk['pre_frames'])
    print("PY Poststimulus frames: ", chunk['post_frames'])

    print("PY Test end")
