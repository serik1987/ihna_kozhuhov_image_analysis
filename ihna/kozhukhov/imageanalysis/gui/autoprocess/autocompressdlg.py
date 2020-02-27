# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.gui.compressiondlg import CompressionDlg
from .trainautocompressdlg import TrainAutocompressDlg


class AutocompressDlg(TrainAutocompressDlg):

    def __init__(self, parent, animal_filter):
        super().__init__(parent, animal_filter, "Autocompress")

    def _open_sub_dlg(self):
        self._sub_dlg = CompressionDlg(self._parent, "Compress",
                                       "Don't compress if target exists",
                                       "Delete decompressed files after compression",
                                       "Compress")
        self._sub_dlg.ShowModal()
