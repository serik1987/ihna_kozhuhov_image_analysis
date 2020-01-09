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
    animals.load()
    animal = animals['c022']
    cases = manifest.CasesList(animal)
    case = cases['0A']
    comp.decompress(case, progress_bar, False, True)
    cases.save()
    print(case)

    '''
    train = files.CompressedFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0A00z")
    train.open()
    print("PY Compressed train was created")

    decompressor = comp._Decompressor(train, "/home/serik1987/vasomotor-oscillations/sample_data/c022z/")
    decompressor.set_progress_bar(progress_bar)
    decompressor.run()
    print("PY Full output file:", decompressor.get_full_output_file())
    '''

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
