from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_DecompressionError as DecompressionError
from ihna.kozhukhov.imageanalysis._imageanalysis import _compression_Compressor as _Compressor
from ihna.kozhukhov.imageanalysis._imageanalysis import _compression_Decompressor as _Decompressor

import os
from ihna.kozhukhov.imageanalysis import ImanError
from ihna.kozhukhov.imageanalysis.manifest import Case
from ihna.kozhukhov.imageanalysis import sourcefiles as sfiles


class NativeDataExistsError(ImanError):
    """
    This exception will be generated when the decompression is started with fail_on_native_exist argument
    set to True and native (decompressed) data have already exist
    """

    def __init__(self):
        super().__init__("The data have already been decompressed")


class NativeDataNotExistsError(ImanError):
    """
    This exception will be generated when you start the compression process and there is no native data to compress
    """

    def __init__(self):
        super().__init__("There is no native data to compress")

class CompressedDataExistsError(ImanError):
    """
    This exception will be generated when you start the compression process with fail_on_compress argument set to
    True and compressed data are present
    """

    def __init__(self):
        super().__init__("The data have already been compressed")


class CompressedDataNotExistsError(ImanError):
    """
    This error will be generated when you try to decompress the file the compressed files are not exist in the
    manifest
    """

    def __init__(self):
        super().__init__("There is no data to decompress")


def decompress(case, progress_bar=None, fail_on_decompress=False, delete_after_decompress=False):
    """
    Provides the data decompression

    Arguments:
        case - the case to be decompressed (instance of ihna.kozhukhov.imageanalysis.manifest.Case)
        Don't forget to save the case list after decompression
        progress_bar - the function that requires float parameter. Total percent completed will be passed to this
        function
        fail_on_decompress - if True, the program will check for native data existence. If the native data was found,
        NativeDataExistsError will be thrown
        delete_after_decompress - the original files will be deleted after the decompression will be completed
    """
    if not isinstance(case, Case):
        raise ValueError("The case is not an instance of Case class")
    if not case.compressed_data_files_exist():
        raise CompressedDataNotExistsError()
    if case.native_data_files_exist() and fail_on_decompress:
        raise NativeDataExistsError()
    input_file = os.path.join(case['pathname'], case['compressed_data_files'][0])
    train = sfiles.CompressedFileTrain(input_file, "traverse")
    train.open()
    output_dir = case['pathname'] + os.path.sep
    decompressor = _Decompressor(train, output_dir)
    if progress_bar is not None:
        decompressor.set_progress_bar(progress_bar)
    decompressor.run()
    output_file = decompressor.get_full_output_file()
    del decompressor
    del train
    output_train = sfiles.StreamFileTrain(output_file, "traverse")
    output_train.open()
    case['native_data_files'] = []
    for output_file in output_train:
        case['native_data_files'].append(output_file.filename)
    del output_train
    if delete_after_decompress:
        file_to_delete = []
        train_to_delete = sfiles.CompressedFileTrain(input_file, "traverse")
        train_to_delete.open()
        for file in train_to_delete:
            fullname = os.path.join(case['pathname'], file.filename)
            file_to_delete.append(fullname)
        del train_to_delete
        for filename in file_to_delete:
            os.remove(filename)
        case['compressed_data_files'] = None


def compress(case, progress_bar=None, fail_on_compress=False, delete_after_compress=False):
    """
    Compresses the file train

    Arguments:
        case - the case to compress (instance of ihna.kozhukhov.imageanalysis.manifest.Case)
        Don't forget to save the case list after compression
        progress_bar - the function like progress_bar(f) where f is float that runs every time the progress is
        achieved. Total percents completed will be passed as an input argument
        fail_on_compress - when True and compressed data exist, the program will be failed.
        delete_after_compress - when True, the native data will be deleted after decompression
    """
    if not isinstance(case, Case):
        raise ValueError("The first argument of the compression function shall be an instance of manifest.Case")
    if not case.native_data_files_exist():
        raise NativeDataNotExistsError()
    if fail_on_compress and case.compressed_data_files_exist():
        raise CompressedDataExistsError()
    working_dir = case['pathname']
    input_train_name = os.path.join(working_dir, case['native_data_files'][0])
    input_train = sfiles.StreamFileTrain(input_train_name, "traverse")
    input_train.open()
    input_file_list = []
    for file in input_train:
        filename = os.path.join(working_dir, file.filename)
        input_file_list.append(filename)
    del file
    compressor = _Compressor(input_train, working_dir + os.path.sep)
    if progress_bar is not None:
        compressor.set_progress_bar(progress_bar)
    compressor.run()
    output_train_name = compressor.get_output_train_name()
    del compressor
    del input_train
    output_train = sfiles.CompressedFileTrain(output_train_name, "traverse")
    output_train.open()
    output_file_list = []
    for output_file in output_train:
        output_file_list.append(output_file.filename)
    del output_file
    case['compressed_data_files'] = output_file_list
    if delete_after_compress:
        for input_file in input_file_list:
            os.remove(input_file)
        case['native_data_files'] = None
    del output_train
    print("Completed")
