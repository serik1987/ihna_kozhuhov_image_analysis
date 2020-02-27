# -*- coding: utf-8

import wx
from ihna.kozhukhov.imageanalysis.gui.compressiondlg import CompressionDlg
from .trainautocompressdlg import TrainAutocompressDlg


class AutodecompressDlg(TrainAutocompressDlg):

    def __init__(self, parent, animal_filter):
        super().__init__(parent, animal_filter, "Autodecompress")

    def _open_sub_dlg(self):
        self._sub_dlg = CompressionDlg(self._parent, "Decompression",
                                       "Don't decompressed file is native data exists",
                                       "Delete compressed files after decompression", "Decompress")
        self._sub_dlg.ShowModal()
