#!/usr/bin/env python3

import ihna.kozhukhov.imageanalysis as iman
import ihna.kozhukhov.imageanalysis.sourcefiles as files

if __name__ == "__main__":
    print("PY Test begin")

    train = files.StreamFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0200", "traverse")
    print("PY Train created")
    train.open()
    print(train)
    print("PY Train ready")

    '''
    try:
        print("PY generating new exception")
        # raise files.CacheSizeError("Sample exception")
        iman.test_exception()
    except iman.ImanError as exc:
        print("PY Error was generated")
        print("PY Exception text:", exc)
        print("PY Exception class:", exc.__class__)
        if isinstance(exc, files.IoError):
            print("PY I/O error")
        if isinstance(exc, files.TrainError):
            print("PY Train error for", exc.train_name)
        if isinstance(exc, files.SourceFileError):
            print("PY Source file error for", exc.file_name)
        if isinstance(exc, files.ChunkError):
            print("PY Chunk error for", exc.chunk_name)
        if isinstance(exc, files.FrameError):
            print("PY Frame error for", exc.frame_number)
    '''

    print("PY Test end")
