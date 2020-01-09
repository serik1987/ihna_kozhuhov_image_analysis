from ihna.kozhukhov.imageanalysis._imageanalysis import _sourcefiles_DecompressionError as DecompressionError
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
        case - the case to be decompressed
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
