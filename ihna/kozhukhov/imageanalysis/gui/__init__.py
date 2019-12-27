# -*- coding: utf-8

import wx
from .MainWindow import MainWindow


def main():
    app = wx.App()
    window = MainWindow()
    window.Show()
    app.MainLoop()
    return 0
