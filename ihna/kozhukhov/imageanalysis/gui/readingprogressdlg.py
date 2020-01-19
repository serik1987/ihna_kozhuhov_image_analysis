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
        print("\033[2K{0}: {1} out of {2}\033[1A".format(message, processed_frames, total_frames))
        if total_frames != self.__maximum:
            processed_frames = processed_frames * self.__maximum // total_frames
        if processed_frames == self.__maximum:
            processed_frames -= 1
        status = self.Update(processed_frames, message)
        return status[0]

    """
    Completely closes the dialog
    """
    def done(self):
        print("\033[2KDone")
        self.Update(self.__maximum)
