# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis import ImagingSignal
from ihna.kozhukhov.imageanalysis.gui.signalviewerdlg import SignalViewerDlg
from ihna.kozhukhov.imageanalysis.gui.autotracereaderdlg import AutotraceReaderDlg
from .trainautoprocessdlg import TrainAutoprocessDlg


class AutotraceDlg(TrainAutoprocessDlg):

    def __init__(self, parent, animal_filter, autodecompress):
        super().__init__(parent, animal_filter, autodecompress, "Autotrace")

    def _open_process_dlg(self, train):
        dlg = AutotraceReaderDlg(self._parent, self._train)
        self._train.close()
        return dlg

    def _set_options(self, case, accumulator):
        roi_name = self._sub_dlg.get_roi_name()
        try:
            roi = case['roi'][roi_name]
        except KeyError:
            raise RuntimeError("ROI with a given name is not present")
        accumulator.set_roi(roi)

    def _get_imaging_class(self):
        return ImagingSignal

    def _get_minor_name(self):
        return "trace(%s)" % self._sub_dlg.get_roi_name()

    def _get_result_view_dlg(self):
        return SignalViewerDlg
