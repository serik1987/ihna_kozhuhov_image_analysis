#!/usr/bin/env python3

import ihna.kozhukhov.imageanalysis as iman
import ihna.kozhukhov.imageanalysis.sourcefiles as files

'''
try:
    iman.test_exception()
except iman.ImanError as e:
    print("Error message: ", e)
    print("Error class name: ", e.__class__.__name__)
    print("Error class: ", e.__class__)
    if isinstance(e, files.IoError):
        print("Instance of files.IoError")
    if isinstance(e, files.TrainError):
        print("Instance of files.TrainError")
        print("Train name: ", e.train_name)
    if isinstance(e, files.SourceFileError):
        print("Instance of files.SourceFileError")
        print("File name: ", e.file_name)
    if isinstance(e, files.ChunkError):
        print("Instance of files.ChunkError")
        print("Chunk name: ", e.chunk_name)
'''

if __name__ == "__main__":
    # train = files.StreamFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0201", "traverse")
    train = files.CompressedFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0A01z", "traverse")
    train.open()
    train.close()
    print("PY Total number of files opened: ", train.file_number)
    print("PY File path: ", train.file_path)
    print("PY Filename: ", train.filename)
    print("PY Frame header size: ", train.frame_header_size)
    print("PY File header size: ", train.file_header_size)
    print("PY Is opened: ", train.is_opened)
    print("PY Stimulation protocol: ", train.experiment_mode)
    print("PY Frame shape: ", train.frame_shape)
    print("PY Frame size: ", train.frame_size)
    print("PY Frame image size: ", train.frame_image_size)
    print("PY Total frame size: ", train.total_frame_size)
    try:
        print("PY Synchronization channel number: ", train.synchronization_channel_number)
        for chan in range(train.synchronization_channel_number):
            print("PY Max value for channel", chan, ": ", train.get_synchronization_channel_max(chan))
    except files.ExperimentModeError:
        print("PY Synchronization channel number is not defined")
    print("PY Total number of frames:", train.total_frames)
    print(train)
