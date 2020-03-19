# -*- coding: utf-8

import wx
from .autoprocessdlg import AutoprocessDlg


class TrainAutocompressDlg(AutoprocessDlg):

    def __init__(self, parent, animal_filter, title):
        super().__init__(parent, animal_filter, title)

    def _process_single_case(self, case, case_box):
        def progress_function(x):
            case_box.progress_function(x, 100.0, "In progress")
        fail_on_decompress = self._sub_dlg.is_fail_on_target_exists()
        delete_after_decompress = self._sub_dlg.is_delete_after_process()
        self._get_processing_function()(case, progress_function, fail_on_decompress, delete_after_decompress)
        print("PY decompression completed")
        if case.get_case_list() is not None:
            case.get_case_list().save()
        else:
            raise RuntimeError("Failure to save the processed data: no case list presented in the case")
        print("PY All cases were saved. Terminating program execution")

    def _get_processing_function(self):
        raise NotImplementedError("TrainAutocompressDlg._get_processing_function")
