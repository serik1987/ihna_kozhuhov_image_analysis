#!/usr/bin/env python3

import imageanalysis.sourcefiles as io

if __name__ == "__main__":
    print(dir(io))
    try:
        raise io.SourceFileError()
    except io.TrainError as exc:
        help(exc)
        print("Exception: {0}".format(exc))
        print("Exception class: {0}".format(exc.__class__.__name__))
        print("Train name: {0}".format(exc.train_name))
        print("File name: {0}".format(exc.file_name))
        raise exc