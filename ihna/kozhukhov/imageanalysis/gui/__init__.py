# -*- coding: utf-8

import wx
from .MainWindow import MainWindow


def main(working_dir):
    app = wx.App()
    window = MainWindow()
    window.open_working_dir(working_dir)
    window.Show()
    wx.MessageDialog(window, "\
(C) Valery Kalatsky, 2003\n\
When using this program reference to the following paper is mandatory:\n\
Kalatsky V.A., Stryker P.S. New Paradigm for Optical Imaging: Temporally\n\
Encoded Maps of Intrinsic Signal. Neuron. 2003. V. 38. N. 4. P. 529-545\n\
(C) Sergei Kozhukhov, 2020\n\
(C) the Institute of Higher Nervous Activity and Neurophysiology,\n\
Russian Academy of Sciences, 2020",
                  "Warning", style=wx.OK|wx.ICON_WARNING|wx.CENTRE).ShowModal()
    app.MainLoop()
    return 0
