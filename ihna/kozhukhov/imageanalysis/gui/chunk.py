# -*- coding: utf-8

import wx


class ChunkViewer(wx.Panel):
    """
    This is the base class for all chunk tabs. Also, if the chunk is unknown or non-recognized,
    such a class corresponds to the chunk

    Initialization:
        viewer = ChunkViewer(parent, chunk)
    where parent is the parent WX window (wx.Notebook instance) and
    chunk is an instance of ihna.kozhukhov.imageanalysis.sourcefiles.Chunk object
    """

    _chunk = None

    def __init__(self, parent, chunk):
        wx.Panel.__init__(self, parent)
        self._chunk = chunk
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_widget = self._get_widgets()
        general_layout.Add(main_widget, 1, wx.EXPAND | wx.ALL, 10)
        self.SetSizer(general_layout)
        self.Layout()

    def _get_widgets(self):
        return wx.StaticText(self, label=str(self._chunk))

    def get_title(self):
        """
        Returns the chunk title (to be substituted to wx.Notebook.AddPage arguments)
        """
        return self._chunk['id']

    @staticmethod
    def new_viewer(parent, chunk):
        if chunk['id'] == "COMP":
            return CompChunkViewer(parent, chunk)
        elif chunk['id'] == "HARD":
            return HardChunkViewer(parent, chunk)
        elif chunk['id'] == "COST":
            return CostChunkViewer(parent, chunk)
        elif chunk['id'] == "DATA":
            return DataChunkViewer(parent, chunk)
        else:
            return ChunkViewer(parent, chunk)


class CompChunkViewer(ChunkViewer):
    """
    A tab page connected to chunk viewer only
    """

    def get_title(self):
        return "Compression properties"

    def _get_widgets(self):
        layout = wx.FlexGridSizer(2, 5, 15)

        label = wx.StaticText(self, label="Size of one single extra pixel")
        layout.Add(label)

        label = wx.StaticText(self, label="{0} bytes".format(self._chunk['compressed_record_size']))
        layout.Add(label)

        label = wx.StaticText(self, label="Size of a single compressed frame")
        layout.Add(label)

        label = wx.StaticText(self, label="{0} bytes".format(self._chunk['compressed_frame_size']))
        layout.Add(label)

        return layout


class HardChunkViewer(ChunkViewer):
    """
    A tab page for the representation of the HARD chunk
    """

    def get_title(self):
        return "Experimental setup"

    def _get_widgets(self):
        layout = wx.FlexGridSizer(2, 5, 15)

        label = wx.StaticText(self, label="Camera name")
        layout.Add(label)

        label = wx.StaticText(self, label=self._chunk['camera_name'])
        layout.Add(label)

        label = wx.StaticText(self, label="Camera type")
        layout.Add(label)

        label = wx.StaticText(self, label=str(self._chunk['camera_type']))
        layout.Add(label)

        label = wx.StaticText(self, label="Physical resolution of CCD chip")
        layout.Add(label)

        label = wx.StaticText(self, label="{0} x {1} pixels".format(self._chunk['resolution_x'],
                                                                    self._chunk['resolution_y']))
        layout.Add(label)

        label = wx.StaticText(self, label="Approximate size of one pixel")
        layout.Add(label)

        label = wx.StaticText(self, label="{0} x {1} um".format(self._chunk['pixel_size_x'],
                                                                self._chunk['pixel_size_y']))
        layout.Add(label)

        label = wx.StaticText(self, label="CCD aperture")
        layout.Add(label)

        label = wx.StaticText(self, label="{0} x {1} um".format(self._chunk['ccd_aperture_x'],
                                                                self._chunk['ccd_aperture_y']))
        layout.Add(label)

        label = wx.StaticText(self, label="Integration time")
        layout.Add(label)

        label = wx.StaticText(self, label="{0} usec".format(self._chunk['integration_time']))
        layout.Add(label)

        label = wx.StaticText(self, label="Interframe time")
        layout.Add(label)

        label = wx.StaticText(self, label="{0} usec".format(self._chunk['interframe_time']))
        layout.Add(label)

        label = wx.StaticText(self, label="Hardware binning")
        layout.Add(label)

        label = wx.StaticText(self, label="{0} x {1}".format(self._chunk['horizontal_hardware_binning'],
                                                             self._chunk['vertical_hardware_binning']))
        layout.Add(label)

        label = wx.StaticText(self, label="Hardware gain")
        layout.Add(label)

        label = wx.StaticText(self, label=str(self._chunk['hardware_gain']))
        layout.Add(label)

        label = wx.StaticText(self, label="Hardware offset")
        layout.Add(label)

        label = wx.StaticText(self, label=str(self._chunk['hardware_offset']))
        layout.Add(label)

        label = wx.StaticText(self, label="CCD size")
        layout.Add(label)

        label = wx.StaticText(self, label="{0} x {1} pixels".format(self._chunk['ccd_size_x'],
                                                                    self._chunk['ccd_size_y']))
        layout.Add(label)

        label = wx.StaticText(self, label="Dynamic range")
        layout.Add(label)

        label = wx.StaticText(self, label=str(self._chunk['dynamic_range']))
        layout.Add(label)

        label = wx.StaticText(self, label="Top lens focal length")
        layout.Add(label)

        label = wx.StaticText(self, label="{0} mm".format(self._chunk['optics_focal_length_top']))
        layout.Add(label)

        label = wx.StaticText(self, label="Bottom lens focal length")
        layout.Add(label)

        label = wx.StaticText(self, label="{0} mm".format(self._chunk['optics_focal_length_bottom']))
        layout.Add(label)

        label = wx.StaticText(self, label="Hardware bits")
        layout.Add(label)

        label = wx.StaticText(self, label="{0:0>32b}".format(self._chunk['hardware_bits']))
        layout.Add(label)

        return layout


class CostChunkViewer(ChunkViewer):
    """
    This is a tab that represents the content of the COST chunk
    """

    def get_title(self):
        return "Stimulation"

    def _get_widgets(self):
        layout = wx.GridBagSizer(5, 15)

        label = wx.StaticText(self, label="Total number of synchronization channels")
        layout.Add(label, pos=(0, 0))

        label = wx.StaticText(self, label=str(self._chunk['synchronization_channel_number']))
        layout.Add(label, pos=(0, 1))

        chan, row = 0, 1
        for max_val in self._chunk['synchronization_channel_max']:
            label = wx.StaticText(self, label="Max value for synchronization channel # {0}".format(chan))
            layout.Add(label, pos=(row, 0))
            label = wx.StaticText(self, label=str(max_val))
            layout.Add(label, pos=(row, 1))
            chan += 1
            row += 1

        label = wx.StaticText(self, label="Total number of stimulus channels")
        layout.Add(label, pos=(row, 0))

        label = wx.StaticText(self, label=str(self._chunk['stimulus_channels']))
        layout.Add(label, pos=(row, 1))
        row += 1

        chan = 0
        for period in self._chunk['stimulus_period']:
            label = wx.StaticText(self, label="Stimulus period for channel # {0}".format(chan))
            layout.Add(label, pos=(row, 0))
            label = wx.StaticText(self, label="{0} ms".format(period))
            layout.Add(label, pos=(row, 1))
            chan += 1
            row += 1

        label = wx.StaticText(self, label="A stimulus channel and a sycnhronization channel \n"
                                          "with the same number are not the same channels")
        layout.Add(label, pos=(row, 0), span=(1, 2))

        return layout


class DataChunkViewer(ChunkViewer):
    """
    A stand-alone page for the DATA chunk
    """

    def get_title(self):
        return "Data"

    def _get_widgets(self):
        return wx.StaticText(self, label="The total size of all the data are {0} bytes. \n"
                             "Use buttons on the right panel to deal with data".format(self._chunk['size']))
