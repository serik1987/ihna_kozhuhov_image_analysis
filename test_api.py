#!/usr/bin/env python3

import matplotlib.pyplot as plt
import ihna.kozhukhov.imageanalysis as iman
import ihna.kozhukhov.imageanalysis.sourcefiles as files
import ihna.kozhukhov.imageanalysis.compression as comp
from ihna.kozhukhov.imageanalysis import manifest


def progress_bar(perc):
    print("{0} percent completed".format(perc))


if __name__ == "__main__":
    print("PY Test begin")

    animals = manifest.Animals("/home/serik1987/vasomotor-oscillations")
    animal = animals['c022']
    cases = manifest.CasesList(animal)
    case = cases['0A']
    comp.compress(case, progress_bar, False, True)
    cases.save()

    '''
    # train = files.StreamFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0201")
    train = files.CompressedFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0A00z")
    print("PY Train created")
    train.open()
    print(train)

    plt.imshow(train[0].body, cmap='gray')
    plt.colorbar()
    plt.show()

    train.clear_cache()
    train.clear_cache()
    '''

    '''
    try:
        print("PY generating new exception")
        # raise comp.DecompressionError("Sample exception")
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
