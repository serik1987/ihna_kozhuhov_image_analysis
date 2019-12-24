"""
This module contains classes responsible for reading the imaging data from the hard disk and writing them into
the hard disk
"""

import os

from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_IoError as IoError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_TrainError as TrainError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_ExperimentModeError as ExperimentModeError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_SynchronizationChannelNumberError \
    as SynchronizationChannelNumberError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_UnsupportedExperimentModeError as \
    UnsupportedExperimentModeError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_SourceFileError as SourceFileError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FrameNumberError as FrameNumberError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_DataChunkError as DataChunkError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_IsoiChunkError as IsoiChunkError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FileSizeError as FileSizeError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_ExperimentChunkError as ExperimentChunkError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FileHeaderError as FileHeaderError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FrameHeaderError as FrameHeaderError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_MapDimensionsError as MapDimensionsError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_DataTypeError as DataTypeError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_CompChunkNotFoundError as CompChunkNotFoundError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FileOpenError as FileOpenError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FileReadError as FileReadError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_ChunkError as ChunkError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_UnsupportedChunkError as UnsupportedChunkError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_ChunkSizeError as ChunkSizeError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_ChunkNotFoundError as ChunkNotFoundError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FileNotOpenedError as FileNotOpenedError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_IsoiChunkNotFoundError as IsoiChunkNotFoundError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FileNotLoadedError as FileNotLoadedError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_DataChunkNotFoundError as DataChunkNotFoundError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_NotAnalysisFileError as NotAnalysisFileError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_NotGreenFileError as NotGreenFileError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_NotInTrainHeadError as NotInTrainHeadError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_NotStreamFileError as NotStreamFileError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_NotCompressedFileError as NotCompressedFileError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FileTrain as FileTrain
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_StreamFileTrain as _StreamFileTrain
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_CompressedFileTrain as _CompressedFileTrain
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_SourceFile as _SourceFile
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_AnalysisSourceFile as _AnalysisSourceFile
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_GreenSourceFile as _GreenSourceFile
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_TrainSourceFile as TrainSourceFile
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_StreamSourceFile as _StreamSourceFile
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FileTrainIterator as FileTrainIterator
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_StreamFileTrainIterator as StreamFileTrainIterator
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_CompressedSourceFile as _CompressedSourceFile
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_CompressedFileTrainIterator as \
    CompressedFileTrainIterator

from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_Chunk as Chunk


class StreamFileTrain(_StreamFileTrain):
    """
    The class allows to perform I/O operations on the file train
    containing the data in the non-compressed mode.
    See definition of the file train on help of the base class

    The object is iterable. Iteration over the object will return
    StreamSourceFile instances corresponding to each file containing
    in the file train
    """

    def __init__(self, filename, traverse_mode):
        """
        Initializes the train

        Arguments:
            filename - full name to the train file
            traverse - defines the object behaviour when filename
            is not at the head of the train. The following options
            are possible:
                "traverse" - find the head of the train
                "exception" - throw an exception
        """
        path, file = os.path.split(filename)
        if len(path) > 0:
            path += os.path.sep
        file_sizes = [1299503468, 1299503468, 1299503468, 1135885676]
        super().__init__(path, file, file_sizes, traverse_mode)


class CompressedFileTrain(_CompressedFileTrain):
    """
    The classs allows to perform I/O operations on the file train
    that contains the data in the compressed mode. Also, the class
    is responsible for compression and decompression

    The class doesn't allow any data processing. In order to process
    the data, please, first, decompress them and then re-open them
    with StreamFileTrain

    The object is iterable. Iterations over object items will return
    CompressedSourceFile instances within the object
    """

    def __init__(self, filename, traverse_mode):
        """
        Initializes the train

        Arguments:
            filename - full name to the train file
            traverse - defines the object behaviour when filename
            is not at the head of the train. The following options
            are possible:
                "traverse" - find the head of the train
                "exception" - throw an exception
        """
        path, file = os.path.split(filename)
        if len(path) > 0:
            path += os.path.sep
        super().__init__(path, file, traverse_mode)


class SourceFile(_SourceFile):
    """
    This class represents basic operations for all source files without
    any extending functionality. It can be applied for a file type checking
    """

    def __init__(self, filename):
        """
        Creates a source file instance.

        Arguments:
            filename - name to the file
        """
        path, file = os.path.split(filename)
        if len(path) > 0:
            path += os.path.sep
        super().__init__(path, file)


class AnalysisSourceFile(_AnalysisSourceFile):
    """
    This class provides I/O operations on a single analysis file.

    The analysis files stores the analysis results
    """

    def __init__(self, filename):
        """
        Creates a new analysis file

        Arguments:
            filename - the full name to the file
        """
        path, file = os.path.split(filename)
        if len(path) > 0:
            path += os.path.sep
        super().__init__(path, file)


class GreenSourceFile(_GreenSourceFile):
    """
    The class provides I/O operations on the files containing so called
    "green maps"
    """

    def __init__(self, filename):
        """
        Creates the file

        Arguments:
            filename - full name to the file
        """
        path, file = os.path.split(filename)
        if len(path) > 0:
            path += os.path.sep
        super().__init__(path, file)


class StreamSourceFile(_StreamSourceFile):
    """
    The class provides I/O operations on a single file that contains native
    data in their uncompressed state

    The class allows to read general information about the experiment from
    the file header. In order to deal with particular frames, please, use
    StreamFileTrain class.
    """

    def __init__(self, filename, traverse_mode):
        """
        Creates new file

        Arguments:
            filename - full name of the file
            traverse_mode - defines the behavior when the filename
            doesn't refer to the head of the train
                'traverse' - find the head of the train and use it instead
                'exception' - throw ihna.kozhukhov.imageanalysis.sourcefiles.NotInTrainHeadError
                'ignore' - just load the file header, don't bear in mind
        """
        path, file = os.path.split(filename)
        if len(path) > 0:
            path += os.path.sep
        super().__init__(path, file, traverse_mode, None)


class CompressedSourceFile(_CompressedSourceFile):
    """
    The class provides I/O operations on a single file that contains native
    data in their compressed state.

    The class itself provides general information about the file. For detailed
    use please, apply CompressedFileTrain
    """

    def __init__(self, filename, traverse_mode):
        """
       Creates new file

       Arguments:
           filename - full name of the file
           traverse_mode - defines the behavior when the filename
           doesn't refer to the head of the train
               'traverse' - find the head of the train and use it instead
               'exception' - throw ihna.kozhukhov.imageanalysis.sourcefiles.NotInTrainHeadError
               'ignore' - just load the file header, don't bear in mind
       """
        path, file = os.path.split(filename)
        if len(path) > 0:
            path += os.path.sep
        super().__init__(path, file, traverse_mode, None)
