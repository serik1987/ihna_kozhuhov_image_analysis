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
    print("PY Test begin")

    train = files.CompressedFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0A01z", "traverse")
    train.open()

    print(train)
    for file in train:
        print(file)

    soft = file.soft
    print("PY Chunk id: ", soft['id'])
    print("PY Chunk size: ", soft['size'])
    print("PY File type: ", soft['file_type'])
    print("PY Recorded date and time: ", soft['date_time_recorded'])
    print("PY User name: ", soft['user_name'])
    print("PY Subject ID: ", soft['subject_id'])
    print("PY Current filename: ", soft['current_filename'])
    print("PY Previous filename: ", soft['previous_filename'])
    print("PY Next filename: ", soft['next_filename'])
    print("PY Data type: ", soft['data_type'])
    print("PY Pixel size: ", soft['pixel_size'])
    print("PY Map X size: ", soft['x_size'])
    print("PY Map Y size: ", soft['y_size'])
    print("PY ROI X position: ", soft['roi_x_position'])
    print("PY ROI Y position: ", soft['roi_y_position'])
    print("PY ROI X size: ", soft['roi_x_size'])
    print("PY ROI Y size: ", soft['roi_y_size'])
    print("PY ROI X position (adjusted): ", soft['roi_x_position_adjusted'])
    print("PY ROI Y position (adjusted): ", soft['roi_y_position_adjusted'])
    print("PY ROI number: ", soft['roi_number'])
    print("PY Temporal binning: ", soft['temporal_binning'])
    print("PY Spatial binning X: ", soft['spatial_binning_x'])
    print("PY Spatial binning Y: ", soft['spatial_binning_y'])
    print("PY Frame header size: ", soft['frame_header_size'])
    print("PY Total frames: ", soft['total_frames'])
    print("PY Number of frames in this file: ", soft['frames_this_file'])
    print("PY Wavelength (nm): ", soft['wavelength'])
    print("PY Filter width (nm): ", soft['filter_width'])

    print("PY Test end")
