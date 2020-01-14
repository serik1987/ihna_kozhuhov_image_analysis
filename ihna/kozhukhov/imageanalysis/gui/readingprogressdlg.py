# -*- coding: utf-8

import wx


class ReadingProgressDialog(wx.ProgressDialog):
    """
    Creates new progress dialog and adapts in for reading
    """

    __maximum = 0

    def __init__(self, parent: wx.TopLevelWindow, title: str, total_frames: int, initial_message: str):
        super().__init__(title, initial_message, maximum=total_frames,
                         style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE | wx.PD_CAN_ABORT, parent=parent)
        self.__maximum = total_frames

    """
    The progress function.
    """
    def progress_function(self, processed_frames, total_frames, message):
        if total_frames != self.__maximum:
            self.__maximum = total_frames
        if processed_frames == total_frames:
            processed_frames -= 1
        status = self.Update(processed_frames, message)
        return status[0]

    """
    Completely closes the dialog
    """
    def done(self):
        self.Update(self.__maximum)
