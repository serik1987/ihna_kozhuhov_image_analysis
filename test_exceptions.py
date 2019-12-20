#!/usr/bin/env python3

from ihna.kozhukhov.imageanalysis.sourcefiles import test
import ihna.kozhukhov.imageanalysis.sourcefiles as files

if __name__ == "__main__":
    L = test.c_api_test()
    print("List length: ", len(L))
    for x in range(1, len(L)):
        print("{0}\t{1}".format(x, L[x]))
    print("Exceptions without C API")
    for object_name in dir(files):
        object = files.__getattribute__(object_name)
        if isinstance(object, type):
            if issubclass(object, Exception):
                if object not in L:
                    print(object)

    n = 0
    while True:
        print("=========================================")
        try:
            test.connection_test(n)
            break
        except Exception as e:
            print("{0}\t{1}".format(e, e.__class__))
            if hasattr(e, "train_name"):
                print("TRAIN ", e.train_name, end = "")
                if hasattr(e, "file_name"):
                    print("\tFILE ", e.file_name, end = "")
                if hasattr(e, "chunk_name"):
                    print("\tCHUNK ", e.chunk_name, end = "")
                print("")
            n += 1
