# -*- coding: utf-8

import os
import wx
from ihna.kozhukhov.imageanalysis.sourcefiles import StreamFileTrain
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
            raise RuntimeError("In order to do this action please, open at least one case containing native data files")
        else:
            if dlg.ShowModal() == wx.ID_CANCEL:
                return False
            self._sub_dlg = dlg
            return True

    def _open_process_dlg(self, train):
        raise NotImplementedError("_open_process_dlg")
