"""
This module contains classes responsible for reading the imaging data from the hard disk and writing them into
the hard disk
"""

import os
import psutil

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
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FileWriteError as FileWriteError
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
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FrameError as FrameError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FrameNotReadError as FrameNotReadError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FrameRangeError as FrameRangeError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_FramChunkNotFoundError as FramChunkNotFoundError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_CompressedFrameReadError as \
    CompressedFrameReadError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_CacheSizeError as CacheSizeError
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_ClearCacheError as ClearCacheError

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
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_SoftChunk as _SoftChunk
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_IsoiChunk as _IsoiChunk
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_CompChunk as _CompChunk
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_CostChunk as _CostChunk
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_DataChunk as _DataChunk
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_EpstChunk as _EpstChunk
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_GreenChunk as _GreenChunk
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_HardChunk as _HardChunk
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_RoisChunk as _RoisChunk
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_SyncChunk as _SyncChunk
from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_IsoiChunkIterator as IsoiChunkIterator

from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_Frame as _Frame


class StreamFileTrain(_StreamFileTrain):
    """
    The class allows to perform I/O operations on the file train
    containing the data in the non-compressed mode.
    See definition of the file train on help of the base class

    The object is iterable. Iteration over the object will return
    StreamSourceFile instances corresponding to each file containing
    in the file train
    """

    def __init__(self, filename, traverse_mode="traverse"):
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
        f = StreamSourceFile(filename, traverse_mode)
        f.open()
        f.load_file_info()
        this_filename = path + f.filename
        size = os.path.getsize(this_filename)
        next_filename = f.soft['next_filename']
        f.close()
        actual_sizes = [size]
        while next_filename != "":
            f = StreamSourceFile(path + next_filename, "ignore")
            f.open()
            f.load_file_info()
            this_filename = path + f.filename
            size = os.path.getsize(this_filename)
            next_filename = f.soft['next_filename']
            f.close()
            actual_sizes.append(size)
        super().__init__(path, file, actual_sizes, traverse_mode)

    def open(self):
        super().open()
        available_memory = psutil.virtual_memory().available // 2
        max_frames = available_memory // self.total_frame_size
        self.capacity = max_frames


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

    def __init__(self, filename, traverse_mode="traverse"):
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


class SoftChunk(_SoftChunk):
    """
    The class represents the data that relates to the single SOFT chunk

    chunk = SoftChunk() will create new chunk. This constructor will not read an existent chunk from the
    hard disk

    Another option to create the SOFT chunk is to use method and properties from SourceFile object
    All SOFT chunks returned by these properties/methods will be read from the hard disk

    Property names for SOFT chunk (type soft["property_name"] to receive the property with such a name):
    id - always "SOFT"
    size - always 256 for (x32 version)
    file_type -'analysis' for analysis file, 'green' for green map, 'stream' for imaging data in non-compressed mode,
    'compressed' for imaging data in compressed mode
    date_time_recorded - date and time where the record has been created
    user_name - name of the user that created this record
    subject_id - name of the subject who created this record
    current_filename - this property shall coincide with the name of the currently opened file
    previous_filename - name of the preceding file in the file train or empty string if the file is at the head of
    the train
    next_filename - name of the following file in the file train or empty string if the file is at the tail of the
    train
    data_type - the ID of the data type
    pixel_size - size of the single pixels on the map, in bytes
    x_size - number of pixels, in horizontal
    y_size - number of pixels, in vertical
    roi_x_position - X position of the upper left corner of ROI (before binning)
    roi_y_position - Y position of the upper left corner of ROI (before binning)
    roi_x_size - X size of the ROI, in pixels (before binning)
    roi_y_size - Y size of the ROI, in pixels (before binning)
    roi_x_position_adjusted - adjusted position of the X ROI (before binning)
    roi_y_position_adjusted - adjusted position of the Y ROI (before binning)
    roi_number - Number of the ROI
    temporal_binning - number of bins for the temporal binning
    spatial_binning_x - number of bins for the spatial binning, on horizontal
    spatial_binning_y - number of bins for the spatial binning, on vertical
    frame_header_size - size of the frame header, in bytes
    total_frames - number of total frames within the whole record
    frames_this_file - number of frames recorded to the opened file
    wavelength - filter wavelength (nm)
    filter_width - the filter width (nm)

    """

    def __init__(self):
        """
        Initializes the chunk
        """
        super().__init__()

class IsoiChunk(_IsoiChunk):
    """
    This is the main chunk that contains all other chunks presented in the file.
    Also, you may use the ISOI chunk in order to traverse through any other chunks

    Formally, you can create an empty ISOI chunk by means of: chunk = IsoiChunk()
    However, this operator is absolutely meaningless. The best way to access the ISOI
    chunk is to use the isoi property of any SourceFile object.

    The object is suitable for navigation over the file header. Use other methods for
    navigation over the DATA chunk (i.e., to navigate through the file body).

    Chunk parameters:
    isoi['id'] will always return the 'ISOI' string
    isoi['size'] will return the total size of the ISOI chunk
    isoi['COST'] will return the any other chunk containing in the chunk header but bot in the chunk body or
    will throw IndexError if such chunk doesn't exist
    """

    def __init__(self):
        super().__init__()


class CompChunk(_CompChunk):
    """
    This chunk presents in compressed files only and contains the compression info

    Chunk property names:
    comp['id'] - always COMP
    comp['size'] = always 28 bytes
    comp['compressed_record_size'] = size of a single extrapixel
    comp['compressed_frame_size'] = size of a single frame (bytes), when this is in compressed mode
    comp['compressed_frame_number'] = number of frames in a certain compressed file
    """

    def __init__(self):
        super().__init__()


class CostChunk(_CostChunk):
    """
    This chunk is presented in the data recorded under continuous stimulation protocol
    and represents the stimulation parameters

    Chunk properties are the following:
    cost['id'] = always 'COST'
    cost['size'] = always 64 bytes
    cost['synchronization_channel_number'] = total number of the synchronization channels
    cost['synchronization_channel_max'] = tuple containing maximum value for each synchronization channel
    cost['stimulus_channels'] = number of stimulus channels (if no synchronization channels were presented)
    cost['stimulus_period'] = tuple containing stimulus period values for each stimulus period
    """

    def __init__(self):
        super().__init__()


class DataChunk(_DataChunk):
    """
    This is the chunk that contains the file body. Specifically, it contains sequence of all
    frames recorded in the current file.

    Chunk have the following properties only:
    data['id'] = always 'DATA'
    data['size'] = size of the file body in uncoompressed mode
    All other data are accessible through the file routines only
    """

    def __init__(self):
        super().__init__()


class EpstChunk(_EpstChunk):
    """
    The chunk contains information about the visual stimulus when episodic stimulation protocol was
    applied. The chunk is absent in the data recorded under continuous stimulation protocol.

    The chunk has the following properties:
    epst['id'] - always "EPST"
    epst['size'] - always 64 bytes
    epst['condition_number'] - total number of stimulus conditions
    epst['repetition_number'] - total number of stimulus repetitions
    epst['randomized'] - True if the stimulus is randomized
    epst['iti_frames'] - Number of intertrial frames
    epst['stimulus_frames'] - Number of stimulus frames
    epst['pre_frames'] - Number of prestimulus frames
    epst['post_frames'] - Number of poststimulus frames
    """

    def __init__(self):
        super().__init__()


class GreenChunk(_GreenChunk):
    """
    The chunk is presented in green files only and stores properties specific for the green chunk
    itself

    At the moment when the program was created, the green chunk is not able to provide I/O operations
    on chunk properties
    """

    def __init__(self):
        super().__init__()


class HardChunk(_HardChunk):
    """
    The chunk contains general properties of the experimental setup

    Among them are:
    hard['id'] - always 'HARD'
    hard['size'] - always 256 bytes
    hard['camera_name'] - the camera name
    hard['camera_type'] - the camera type
    hard['resolution_x'] - resolution on X
    hard['resolution_y'] - resolution on Y
    hard['pixel_size_x'] - pixel size on X, in um (rounded to integer)
    hard['pixel_size_y'] - pixel size on Y, in um (rounded to integer)
    hard['ccd_aperture_x'] - CCD aperture on X, in um (rounded to integer)
    hard['ccd_aperture_y'] - CCD aperture on Y, in um (rounded to integer)
    hard['integration time'] - the integration time in microseconds
    hard['interframe_time'] - the interframe time in microseconds
    hard['vertical_hardware_binning'] - vertical hardware binning
    hard['horizontal_hardware_binning'] - horizontal hardware binning
    hard['hardware_gain'] - the hardware gain
    hard['hardware_offset'] - the hardware offset
    hard['ccd_size_x'] - hardware binned CCD X size
    hard['ccd_size_y'] - hardware binned CCD Y size
    hard['dynamic_range'] - the dynamic range
    hard['optics_focal_length_top'] - Top lens focal length (the one closer to the camera), millimeters
    hard['optics_focal_length_bottom'] - Bottom lens focal length, millimeters
    hard['hardware_bits'] - the hardware bits
    """

    def __init__(self):
        super().__init__()


class RoisChunk(_RoisChunk):
    """
    This is the rarely used chunk which is mostly non-implemented
    """

    def __init__(self):
        super().__init__()


class SyncChunk(_SyncChunk):
    """
    This is the rarely used which which is mostly non-implemented
    """

    def __init__(self):
        super().__init__()


def get_file_info(selected_files, discarded_files=None, progress_bar=None):
    """
    Reads file header for each files in the given list and returns information about this file
    as an output

    Input arguments:
        selected_files - list of str that contains absolute or relative file names. This function will read the file
        header belonging to each file that is presented within the selected_files list but absent in the
        discarded_files list

        discarded_files - list of files which headers will not be looked for. None means that there is no such files

        progress_bar - is this argument is a function like f(i, N) such function will be called within the each 10 files
        total number of files processed will be referred to i and total files to process will be N

    Output arguments:
        a tuple containing two values.
        The first value is list of all files that can be opened and processed. If the file is a part of the file train,
        the list item refers to the train head while all other files within the train will be ignored.
            'filename' - full name of the file. In case of train, the full name of the file head
            'filetype' - the main file type, like: 'analysis' - the analysis file; 'green' - the green file
            The following key presents in the train file only:
            'tail_files' - list of the files that are within the same file train as considering file but not at the
            head of the train.

        The second value is the list of all files that are incorrect and hence will be discarded. Files that are within
        the file train but not within the head of the train will not be included in both lists
    """
    if discarded_files is None:
        discarded_files = []
    done_files = []
    valid_files = []
    invalid_files = []
    idx = 0
    for filename in selected_files:
        if filename in discarded_files or filename in done_files:
            continue
        is_valid = False
        try:
            analysis_file = AnalysisSourceFile(filename)
            analysis_file.open()
            analysis_file.load_file_info()
            analysis_file.close()
            valid_files.append({
                "filename": filename,
                "filetype": "analysis"
            })
            is_valid = True
            done_files.append(filename)
            # print("{0}\tanalysis".format(filename))
        except IoError:
            pass
        try:
            green_file = GreenSourceFile(filename)
            green_file.open()
            green_file.load_file_info()
            green_file.close()
            valid_files.append({
                "filename": filename,
                filetype: "green"
            })
            is_valid = True
            done_files.append(filename)
            # print("{0}\tgreen".format(filename))
        except IoError:
            pass
        try:
            compressed_file_train = CompressedFileTrain(filename, "traverse")
            compressed_file_train.open()
            compressed_file_train.close()
            head_filename = compressed_file_train.file_path + compressed_file_train.filename
            valid_file = {
                "filename": head_filename,
                "filetype": "compressed",
                "tail_files": []
            }
            # print("{0}\tcompressed".format(filename), end="\t")
            for file in compressed_file_train:
                if file.full_name != head_filename:
                    valid_file['tail_files'].append(file.full_name)
                    # print(file.filename, end=" ")
                done_files.append(file.full_name)
            valid_files.append(valid_file)
            is_valid = True
            # print("")
        except IoError:
            pass
        try:
            stream_file_train = StreamFileTrain(filename, "traverse")
            stream_file_train.open()
            stream_file_train.close()
            head_filename = stream_file_train.file_path + stream_file_train.filename
            valid_file = {
                "filename": head_filename,
                "filetype": "stream",
                "tail_files": []
            }
            # print("{0}\tstream".format(head_filename), end="\t\t")
            for file in stream_file_train:
                if file.full_name != head_filename:
                    valid_file['tail_files'].append(file.full_name)
                    # print(file.filename, end=" ")
                done_files.append(file.full_name)
            valid_files.append(valid_file)
            is_valid = True
            # print("")
        except IoError:
            pass
        if not is_valid:
            invalid_files.append(filename)
            done_files.append(filename)
            # print("{0}\tINVALID".format(filename))
        idx += 1
        if idx % 10 == 0 and progress_bar is not None:
            progress_bar(idx, len(selected_files))

    return valid_files, invalid_files
