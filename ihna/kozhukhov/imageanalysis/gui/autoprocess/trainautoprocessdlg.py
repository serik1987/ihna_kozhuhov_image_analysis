# -*- coding: utf-8

import os
import wx
from ihna.kozhukhov.imageanalysis.compression import decompress
from ihna.kozhukhov.imageanalysis.sourcefiles import StreamFileTrain
from ihna.kozhukhov.imageanalysis.manifest import Case
from .autoprocessdlg import AutoprocessDlg


class TrainAutoprocessDlg(AutoprocessDlg):

    _autodecompress = None
    _train = None

    def __init__(self, parent, animal_filter, autodecompress, title):
        super().__init__(parent, animal_filter, title)
        self._autodecompress = autodecompress

    def _open_sub_dlg(self):
        dlg = None
        for case in self._animal_filter:
            filename = "[unknown]"
            try:
                if case['native_data_files'] is None:
                    raise RuntimeError("No native data")
                filename = os.path.join(case['pathname'], case['native_data_files'][0])
                self._train = StreamFileTrain(filename)
                self._train.open()
                dlg = self._open_process_dlg(self._train)
                if dlg is None:
                    raise ValueError("_open_process_dlg shall return the dialog")
                print("PY Success in opening {0}".format(filename))
                break
            except Exception as err:
                print("PY Opening {0} failed due to: {1}".format(filename, err))
        self._animal_filter.reset_iteration()
        if dlg is None:
            raise RuntimeError("In order to do this action please, open at least one case containing native data files"
                               ". This case shall be included in autoprocess and autocompress")
        else:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return False
            self._sub_dlg = dlg
            return True

    def _open_process_dlg(self, train):
        raise NotImplementedError("_open_process_dlg")

    def _process_single_case(self, case: Case, case_box):
        def progress_function(x):
            case_box.progress_function(x, 100, "Decompression")

        pathname = case['pathname']
        if case.native_data_files_exist():
            is_decompressing = False
            filename = case['native_data_files'][0]
        else:
            if self._autodecompress:
                is_decompressing = True
                decompress(case, progress_function, False, False)
                filename = case['native_data_files'][0]
            else:
                raise RuntimeError("The data are compressed")
        fullname = os.path.join(pathname, filename)

        train = StreamFileTrain(fullname)
        train.open()
        accumulator = self._sub_dlg.create_accumulator(train)
        try:
            self._set_options(case, accumulator)
            accumulator.set_progress_bar(case_box)
            accumulator.accumulate()

            major_name = "%s_%s%s%s" % (
                case.get_animal_name(),
                self._sub_dlg.get_prefix_name(),
                case['short_name'],
                self._sub_dlg.get_postfix_name()
            )
            result_data = self._get_imaging_class()(accumulator, major_name)
            result_data.get_features().update(self._sub_dlg.get_options())
            result_data.get_features()['minor_name'] = self._get_minor_name()
            self._set_result_options(result_data, self._sub_dlg)
        except Exception as err:
            del accumulator
            train.close()
            train.clear_cache()
            del train
            if is_decompressing:
                case.delete_native_files()
            raise err
        del accumulator
        train.close()
        train.clear_cache()
        del train
        if is_decompressing:
            case.delete_native_files()

        if self._sub_dlg.is_save_npz():
            result_data.save_npz(pathname)
            if self._sub_dlg.is_add_to_manifest():
                case.add_data(result_data)
                case.get_case_list().save()
        if self._sub_dlg.is_save_mat():
            result_data.save_mat(pathname)
        if self._sub_dlg.is_save_png():
            result_dlg = self._get_result_view_dlg()(self, result_data)
            result_dlg.save_png(pathname)

    def _set_options(self, case, accumulator):
        pass

    def _set_result_options(self, result_data, sub_dlg):
        pass

    def _get_imaging_class(self):
        raise NotImplementedError("TrainAutoProcess._get_imaging_class")

    def _get_minor_name(self):
        raise NotImplementedError("TrainAutoProcess._get_minor_name")

    def _get_result_view_dlg(self):
        raise NotImplementedError("TrainAutoProcess._get_result_view_dlg")
