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