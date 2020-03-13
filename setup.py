# -*- coding: utf-8

"""
Copyright (C) 2003 Valery Kalatsky
Copyright (C) 2020 Sergei A. Kozhukhov
Copyright (C) 2020 the Institute of Higher Nervous Activity, Russian Academy of Sciences
"""

import os
import sys
import platform
from distutils.core import setup, Extension
try:
    import numpy
except ImportError:
    print("The NUMPY package has not been found. The following command will fix your problem:")
    print("")
    print("pip install numpy")
    print("")
if platform.system().lower() == "windows":
    extra_compile_args = ["/std:c++17"]
else:
    os.environ['CC'] = "g++"
    extra_compile_args = ["-std=c++17", "-pedantic"]

if sys.version[0] != '3':
    print("Please, run this setup program under python3")
    sys.exit(-1)

imageanalysis = Extension("ihna.kozhukhov.imageanalysis._imageanalysis",
                          sources=[
                              "imageanalysis.cpp",
                              "cpp/source_files/SourceFile.cpp",
                              "cpp/source_files/ChunkHeader.cpp",
                              "cpp/source_files/Chunk.cpp",
                              "cpp/source_files/FramChunk.cpp",
                              "cpp/source_files/FramCostChunk.cpp",
                              "cpp/source_files/FramEpstChunk.cpp",
                              "cpp/source_files/IsoiChunk.cpp",
                              "cpp/source_files/HardChunk.cpp",
                              "cpp/source_files/SoftChunk.cpp",
                              "cpp/source_files/CostChunk.cpp",
                              "cpp/source_files/EpstChunk.cpp",
                              "cpp/source_files/GreenChunk.cpp",
                              "cpp/source_files/DataChunk.cpp",
                              "cpp/source_files/SyncChunk.cpp",
                              "cpp/source_files/RoisChunk.cpp",
                              "cpp/source_files/CompChunk.cpp",
                              "cpp/source_files/AnalysisSourceFile.cpp",
                              "cpp/source_files/GreenSourceFile.cpp",
                              "cpp/source_files/TrainSourceFile.cpp",
                              "cpp/source_files/StreamSourceFile.cpp",
                              "cpp/source_files/CompressedSourceFile.cpp",
                              "cpp/source_files/FileTrain.cpp",
                              "cpp/source_files/StreamFileTrain.cpp",
                              "cpp/source_files/CompressedFileTrain.cpp",
                              "cpp/source_files/Frame.cpp",
                              "cpp/compression/BaseCompressor.cpp",
                              "cpp/compression/Compressor.cpp",
                              "cpp/compression/Decompressor.cpp",
                              "cpp/tracereading/PixelListItem.cpp",
                              "cpp/tracereading/TraceReader.cpp",
                              "cpp/synchronization/Synchronization.cpp",
                              "cpp/synchronization/NoSynchronization.cpp",
                              "cpp/synchronization/QuasiStimulusSynchronization.cpp",
                              "cpp/synchronization/QuasiTimeSynchronization.cpp",
                              "cpp/synchronization/ExternalSynchronization.cpp",
                              "cpp/isolines/Isoline.cpp",
                              "cpp/isolines/NoIsoline.cpp",
                              "cpp/isolines/LinearFitIsoline.cpp",
                              "cpp/isolines/TimeAverageIsoline.cpp",
                              "cpp/tracereading/TraceReaderAndCleaner.cpp",
                              "cpp/misc/LinearFit.cpp",
                              "cpp/accumulators/Accumulator.cpp",
                              "cpp/accumulators/TraceAutoReader.cpp",
                              "cpp/accumulators/FrameAccumulator.cpp",
                              "cpp/accumulators/MapPlotter.cpp",
                              "cpp/accumulators/MapFilter.cpp"
                          ],
                          include_dirs=[numpy.get_include()],
                          extra_compile_args=extra_compile_args
                          )

setup(
    name="ihna.kozhukhov.imageanalysis",
    version='1.0',
    description="Processing of intrinsic-signal imaging data revealed under continuous stimulation",
    author="Sergei A. Kozhukhov, Valery Kalatsky",
    author_email="serik1987@gmail.com",
    url="https://github.com/serik1987/ihna_kozhuhov_image_analysis",
    long_description="The package is used to process the intrinsic-signal imaging data recorded under continuous\n"
    "stimulation by the method developed by V. Kalatsky. The modules within the package open the files generated by\n"
    "V. Kalatsky's data acquisition setup and plots traces and averaged maps. Instruments for the following\n"
    "processing of traces and averaged maps are also provided",
    license="(C) Valery Kalatsky, 2003\n"
    "(C) Sergei A. Kozhukhov, 2020\n"
    "(C) the Institute of Higher Nervous Activity and Neurophysiology, Russian Academy of Sciences",
    packages=[
        "ihna",
        "ihna.kozhukhov",
        "ihna.kozhukhov.imageanalysis",
        "ihna.kozhukhov.imageanalysis.accumulators",
        "ihna.kozhukhov.imageanalysis.compression",
        "ihna.kozhukhov.imageanalysis.gui",
        "ihna.kozhukhov.imageanalysis.gui.autoprocess",
        "ihna.kozhukhov.imageanalysis.gui.isolines",
        "ihna.kozhukhov.imageanalysis.gui.mapfilterdlg",
        "ihna.kozhukhov.imageanalysis.gui.synchronization",
        "ihna.kozhukhov.imageanalysis.isolines",
        "ihna.kozhukhov.imageanalysis.manifest",
        "ihna.kozhukhov.imageanalysis.mapprocessing",
        "ihna.kozhukhov.imageanalysis.sourcefiles",
        "ihna.kozhukhov.imageanalysis.synchronization",
        "ihna.kozhukhov.imageanalysis.tracereading"
    ],
    ext_modules=[imageanalysis],
    requires=[
        "psutil",
        "numpy",
        "matplotlib",
        "scipy",
        "wxpython",
        "pypng"
    ],
    provides=["ihna_kozhukhov_iman"],
    scripts=["iman"]
)

print("This action has been accomplished successfully!")
