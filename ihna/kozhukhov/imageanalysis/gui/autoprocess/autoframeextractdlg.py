# -*- coding: utf-8

import os
from png import Writer
import numpy as np
from .autoprocessdlg import AutoprocessDlg
from ihna.kozhukhov.imageanalysis.sourcefiles import StreamFileTrain, CompressedFileTrain


class AutoFrameExtractDlg(AutoprocessDlg):

    def __init__(self, parent, animal_filter):
        super().__init__(parent, animal_filter, "Frame extraction")

    def _process_single_case(self, case, case_box):
        case_box.progress_function(0, 1, "Frame reading")
        pathname = case['pathname']
        filename = None
        if case['native_data_files'] is None:
            if case['compressed_data_files'] is None:
                raise RuntimeError("Native data not found")
            else:
                filename = case['compressed_data_files'][0]
                TrainClass = CompressedFileTrain
        else:
            filename = case['native_data_files'][0]
            TrainClass = StreamFileTrain
        fullname = os.path.join(pathname, filename)

        train = TrainClass(fullname)
        train.open()
        frame_data = train[0].body
        frame_data = frame_data / frame_data.max()
        frame_data *= 256
        frame_data = np.array(frame_data, dtype=np.uint8)
        height, width = frame_data.shape

        writer = Writer(width, height, greyscale=True, bitdepth=8)
        output_file = "%s_%s.png" % (case.get_animal_name(), case['short_name'])
        output_file = os.path.join(pathname, output_file)
        f = open(output_file, "wb")
        writer.write(f, frame_data)
        f.close()
