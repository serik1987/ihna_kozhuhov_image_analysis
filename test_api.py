#!/usr/bin/env python3

import ihna.kozhukhov.imageanalysis.sourcefiles as files

if __name__ == "__main__":
    print("PY Test begin")

    train = files.CompressedFileTrain("/home/serik1987/vasomotor-oscillations/sample_data/c022z/T_1BF.0A00z", "traverse")
    train.open()

    for file in train:
        print(file.filename)

    chunk = file.isoi['hard']
    print(chunk)
    print("PY Chunk ID: ", chunk["id"])
    print("PY Chunk size: ", chunk['size'])
    print("PY Camera name: ", chunk['camera_name'])
    print("PY Camera type: ", chunk['camera_type'])
    print("PY Resolution X: ", chunk['resolution_x'])
    print("PY Resolution Y: ", chunk['resolution_y'])
    print("PY Pixel size X: ", chunk['pixel_size_x'])
    print("PY Pixel size Y: ", chunk['pixel_size_y'])
    print("PY CCD aperture X: ", chunk['ccd_aperture_x'])
    print("PY CCD aperture Y: ", chunk['ccd_aperture_y'])
    print("PY Integration time: ", chunk['integration_time'])
    print("PY Interframe time: ", chunk['interframe_time'])
    print("PY Vertical hardware binning: ", chunk['vertical_hardware_binning'])
    print("PY Horizontal hardware binning: ", chunk['horizontal_hardware_binning'])
    print("PY Hardware gain: ", chunk['hardware_gain'])
    print("PY Hardware offset: ", chunk['hardware_offset'])
    print("PY Hardware binned CCD X size: ", chunk['ccd_size_x'])
    print("PY Hardware binned CCD Y size: ", chunk['ccd_size_y'])
    print("PY Dynamic range: ", chunk['dynamic_range'])
    print("PY Top lens focal length (the one closer to the camera), millimeters: ", chunk['optics_focal_length_top'])
    print("PY Bottom lens focal length, millimeters: ", chunk['optics_focal_length_bottom'])
    print("PY Hardware bits: ", chunk['hardware_bits'])

    print("PY Test end")
