# -*- coding: utf-8

import os.path
import time
import wx
import scipy
from ihna.kozhukhov.imageanalysis import ImagingMap, ImagingSignal
import ihna.kozhukhov.imageanalysis.sourcefiles as sfiles
from ihna.kozhukhov.imageanalysis import compression, accumulators
from ihna.kozhukhov.imageanalysis.tracereading import TraceProcessor
from .chunk import ChunkViewer
from .frameviewer import FrameViewer
from .compressiondlg import CompressionDlg
from .traceanalysispropertiesdlg import TraceAnalysisPropertiesDlg
from .readingprogressdlg import ReadingProgressDialog
from .tracesdlg import TracesDlg
from .finaltracesdlg import FinalTracesDlg
from .mapplotterdlg import MapPlotterDlg
from .mapviewerdlg import MapViewerDlg
from .signalviewerdlg import SignalViewerDlg
from .mapfilterdlg.basicwindow import BasicWindow as MapFilterDlg


class NativeDataManager(wx.Dialog):
    """
    This class is used to open the native data manager

    Usage:
    NativeDataManager(parent, case).ShowModal()
    where:
    parent - the parent window (instance of wx.Frame or wx.Dialog)
    case - an instance of ihna.kozhukhov.imageanalysis.manifest.Case
    """

    __case = None
    __train = None
    __property = None
    __case_name = None
    __TrainClass = None

    def __init__(self, parent, case, case_name="undefined"):
        self.__case = case
        self.__case_name = case_name
        pathname = self.__case['pathname']
        if self.__case.compressed_data_files_exist():
            self.__property = 'compressed_data_files'
            filename = self.__case['compressed_data_files'][0]
            fullname = os.path.join(pathname, filename)
            self.__train = sfiles.CompressedFileTrain(fullname, "traverse")
            self.__TrainClass = sfiles.CompressedFileTrain
        elif self.__case.native_data_files_exist():
            self.__property = 'native_data_files'
            filename = self.__case['native_data_files'][0]
            fullname = os.path.join(pathname, filename)
            self.__train = sfiles.StreamFileTrain(fullname, "traverse")
            self.__TrainClass = sfiles.StreamFileTrain
        else:
            raise RuntimeError("Error in opening native or compressed files")
        self.__train.open()
        title = "Native data manager: " + case_name
        super().__init__(parent, title=title, size=(800, 600))
        main_panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.HORIZONTAL)

        left_panel = self.__create_left_panel(main_panel)
        main_layout.Add(left_panel, 6, wx.RIGHT | wx.EXPAND, 5)

        right_panel = self.__create_right_panel(main_panel)
        main_layout.Add(right_panel, 2, wx.EXPAND, 0)

        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        main_panel.SetSizer(general_layout)
        self.Centre()

        self.__train.close()

    def __create_left_panel(self, main_panel):
        left_panel = wx.Notebook(main_panel, style=wx.BK_DEFAULT)

        general_properties_page = self.__create_general_properties(left_panel)
        left_panel.AddPage(general_properties_page, "General Properties")

        for file in self.__train:
            break

        for chunk in file.isoi:
            if chunk['id'] != "SOFT":
                viewer = ChunkViewer.new_viewer(left_panel, chunk)
                left_panel.AddPage(viewer, viewer.get_title())

        return left_panel

    def __create_general_properties(self, parent):
        general_properties_page = wx.Panel(parent)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.FlexGridSizer(2, 5, 15)

        label = wx.StaticText(general_properties_page, label="Number of files in the train")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=str(self.__train.file_number))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Total number of frames")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=str(self.__train.total_frames))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Frame dimensions")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} x {1} pixels".format(self.__train.frame_shape[1],
                                                                                       self.__train.frame_shape[0]))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Total number of pixels in the frame")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=str(self.__train.frame_size))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Frame body size")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} bytes".format(self.__train.frame_image_size))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Frame header size")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} bytes".format(self.__train.frame_header_size))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Total frame size (header + body)")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} bytes".format(self.__train.total_frame_size))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="File header size")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} bytes".format(self.__train.file_header_size))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Experiment mode")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=self.__train.experiment_mode)
        main_layout.Add(label)

        for file in self.__train:
            break
        soft = file.soft

        label = wx.StaticText(general_properties_page, label="Date and time of the record")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=soft['date_time_recorded'])
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="User name")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=soft['user_name'])
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Subject ID")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=soft['subject_id'])
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Pixel size")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} bytes".format(soft['pixel_size']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="ROI position (before binning)")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="({0}, {1}) pixels".format(soft['roi_x_position'],
                                                                                        soft['roi_y_position']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="ROI position (adjusted, before binning)")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="({0}, {1}) pixels".format(soft['roi_x_position_adjusted'],
                                                                                        soft['roi_y_position_adjusted']
                                                                                        ))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="ROI size (before binning)")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} x {1} pixels".format(soft['roi_x_size'],
                                                                                       soft['roi_y_size']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="ROI number")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=str(soft['roi_number']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Number of bins for temporal binning")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label=str(soft['temporal_binning']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Spatial binning")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} x {1} pixels".format(soft['spatial_binning_x'],
                                                                                       soft['spatial_binning_y']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Wavelength")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} nm".format(soft['wavelength']))
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="Filter width")
        main_layout.Add(label)

        label = wx.StaticText(general_properties_page, label="{0} nm".format(soft['filter_width']))
        main_layout.Add(label)

        general_layout.Add(main_layout, 1, wx.EXPAND | wx.ALL, 10)
        general_properties_page.SetSizer(general_layout)
        general_properties_page.Layout()
        return general_properties_page

    def __create_right_panel(self, main_panel):
        right_panel = wx.BoxSizer(wx.VERTICAL)
        compression_box = wx.StaticBoxSizer(wx.HORIZONTAL, main_panel, label="Compression")

        compress_button = wx.Button(main_panel, label="Compress")
        self.Bind(wx.EVT_BUTTON, lambda event: self.compress(), compress_button)
        compression_box.Add(compress_button, 1, wx.EXPAND | wx.RIGHT, 5)
        if self.__case['native_data_files'] is None:
            compress_button.Enable(False)
            processing_enabled = False
        else:
            processing_enabled = True

        decompress_button = wx.Button(main_panel, label="Decompress")
        self.Bind(wx.EVT_BUTTON, lambda event: self.decompress(), decompress_button)
        compression_box.Add(decompress_button, 1, wx.EXPAND)
        if self.__case['compressed_data_files'] is None:
            decompress_button.Enable(False)

        right_panel.Add(compression_box, 0, wx.BOTTOM | wx.EXPAND, 5)
        processing_box = wx.StaticBoxSizer(wx.VERTICAL, main_panel, label="Processing")
        processing_box_layout = wx.GridBagSizer(5, 5)

        frame_view_button = wx.Button(main_panel, label="Frame view")
        self.Bind(wx.EVT_BUTTON, lambda event: self.frame_view(), frame_view_button)
        processing_box_layout.Add(frame_view_button, pos=(0, 0), flag=wx.EXPAND)

        map_filter_button = wx.Button(main_panel, label="Map filter")
        self.Bind(wx.EVT_BUTTON, lambda event: self.map_filter(), map_filter_button)
        processing_box_layout.Add(map_filter_button, pos=(0, 1), flag=wx.EXPAND)

        averaged_maps_button = wx.Button(main_panel, label="Averaged maps")
        self.Bind(wx.EVT_BUTTON, lambda event: self.get_averaged_maps(), averaged_maps_button)
        averaged_maps_button.Enable(processing_enabled)
        processing_box_layout.Add(averaged_maps_button, pos=(1, 0), flag=wx.EXPAND)

        trace_analysis_button = wx.Button(main_panel, label="Trace analysis")
        self.Bind(wx.EVT_BUTTON, lambda event: self.trace_analysis(), trace_analysis_button)
        trace_analysis_button.Enable(processing_enabled)
        processing_box_layout.Add(trace_analysis_button, pos=(1, 1), flag=wx.EXPAND)

        processing_box.Add(processing_box_layout, 0, wx.EXPAND)
        right_panel.Add(processing_box, 0, wx.EXPAND)
        return right_panel

    def close(self):
        """
        Closes the file opened by this dialog and clears cache from this
        """
        self.__train.close()
        self.__train = None

    def compress(self):
        dlg = CompressionDlg(self, "Compression properties",
                             "Don't compress the data if compressed files already exists",
                             "Delete original files after compression", "Compress")
        if dlg.ShowModal() == wx.ID_CANCEL:
            return
        fail_on_exist = dlg.is_fail_on_target_exists()
        delete_after_compression = dlg.is_delete_after_process()
        progress_box = wx.ProgressDialog("Compression", "Compression is in progress", maximum=100, parent=self,
                                         style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE)
        progress_box.Update(0)
        try:
            compression.compress(self.__case, lambda perc: progress_box.Update(int(round(perc))),
                                 fail_on_exist, delete_after_compression)
        except Exception as err:
            error_box = wx.MessageDialog(self, str(err), "Compression", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            error_box.ShowModal()
        progress_box.Destroy()
        self.Close()

    def decompress(self):
        dlg = CompressionDlg(self, "Decompression properties",
                             "Don't decompress if native data exists",
                             "Delete compressed files after decompression", "Decompress")
        if dlg.ShowModal() == wx.ID_CANCEL:
            return
        fail_on_exist = dlg.is_fail_on_target_exists()
        delete_after_decompression = dlg.is_delete_after_process()
        progress_box = wx.ProgressDialog("Decompression", "Decompression is in progress", maximum=100, parent=self,
                                         style=wx.PD_APP_MODAL | wx.PD_AUTO_HIDE)
        progress_box.Update(0)
        try:
            compression.decompress(self.__case, lambda perc: progress_box.Update(int(round(perc))),
                                   fail_on_exist, delete_after_decompression)
        except Exception as err:
            error_box = wx.MessageDialog(self, str(err), "Decompression", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            error_box.ShowModal()
        progress_box.Destroy()
        self.Close()

    def frame_view(self):
        try:
            pathname = self.__case['pathname']
            filename = self.__case[self.__property][0]
            fullname = os.path.join(pathname, filename)
            self.__train = self.__TrainClass(fullname)
            self.__train.open()

            viewer = FrameViewer(self, self.__train, self.__case_name)
            viewer.ShowModal()
            viewer.close()
            self.__train.close()
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), caption="Frame viewer", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def get_averaged_maps(self):
        self.__train.close()

        try:
            pathname = self.__case['pathname']
            filename = self.__case['native_data_files'][0]
            fullname = os.path.join(pathname, filename)
            train = sfiles.StreamFileTrain(fullname)
            train.open()
            map_plotter_dlg = MapPlotterDlg(self, train)
            if map_plotter_dlg.ShowModal() == wx.ID_CANCEL:
                map_plotter_dlg.close()
                return
            try:
                plotter = map_plotter_dlg.create_accumulator()
            except Exception as err:
                map_plotter_dlg.close()
                raise err
            map_plotter_dlg.close()
            progress_dlg = ReadingProgressDialog(self, "Map averaging", 1000, "Map averaging")
            try:
                plotter.set_progress_bar(progress_dlg)
                plotter.accumulate()
            except Exception as err:
                progress_dlg.done()
                raise err
            progress_dlg.done()
            animal_name = self.__case.get_animal_name()
            prefix_name = map_plotter_dlg.get_prefix_name()
            postfix_name = map_plotter_dlg.get_postfix_name()
            short_name = self.__case["short_name"]
            major_name = "%s_%s%s%s" % (animal_name, prefix_name, short_name, postfix_name)
            result_map = ImagingMap(plotter, major_name)
            MapViewerDlg(self, result_map).ShowModal()
        except Exception as err:
            print("Error class:", err.__class__.__name__)
            print("Error message:", err)
            dlg = wx.MessageDialog(self, str(err), "Averaged maps", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def map_filter(self):
        self.__train.close()

        try:
            pathname = self.__case['pathname']
            filename = self.__case['native_data_files'][0]
            fullname = os.path.join(pathname, filename)
            train = sfiles.StreamFileTrain(fullname)
            train.open()
            map_filter_dlg = MapFilterDlg(self, train)
            if map_filter_dlg.ShowModal() == wx.ID_CANCEL:
                map_filter_dlg.close()
                return
            try:
                map_filter = map_filter_dlg.create_accumulator()
            except Exception as err:
                map_filter_dlg.close()
                raise err
            map_filter_dlg.close()
            progress_dlg = ReadingProgressDialog(self, "Map filtration", 1000, "Map filtration")
            try:
                map_filter.set_progress_bar(progress_dlg)
                map_filter.accumulate()
            except Exception as err:
                progress_dlg.done()
                raise err
            progress_dlg.done()
        except Exception as err:
            print("Error class:", err.__class__.__name__)
            print("Error message:", err)
            dlg = wx.MessageDialog(self, str(err), "Averaged maps", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            dlg.ShowModal()

    def trace_analysis(self):
        self.__train.close()

        try:
            pathname = self.__case['pathname']
            filename = self.__case['native_data_files'][0]
            fullname = os.path.join(pathname, filename)
            train = sfiles.StreamFileTrain(fullname)
            train.open()

            properties_dlg = TraceAnalysisPropertiesDlg(self, train, self.__case['roi'])
            if properties_dlg.ShowModal() == wx.ID_CANCEL:
                properties_dlg.close()
                del train
                return

            if properties_dlg.is_autoaverage():
                self.__trace_analysis_auto(train, properties_dlg)
            else:
                self.__trace_analysis_manual(train, properties_dlg)
            properties_dlg.close()
            if train.is_opened:
                train.close()
            del train
        except Exception as err:
            dlg = wx.MessageDialog(self, str(err), caption="Trace analysis", style=wx.OK | wx.CENTRE | wx.ICON_ERROR)
            print("Exception class:", err.__class__.__name__)
            print("Exception message:", str(err))
            dlg.ShowModal()

    def __trace_analysis_auto(self, train, properties_dlg):
        sync = properties_dlg.create_synchronization()
        isoline = properties_dlg.create_isoline(sync)
        reader = accumulators.TraceAutoReader(isoline)
        roi_name = properties_dlg.get_roi_name()
        roi = self.__case['roi'][roi_name]
        reader.set_roi(roi)
        progress_dlg = ReadingProgressDialog(self, "Trace analysis", 1000, "Preparing")
        reader.set_progress_bar(progress_dlg)
        try:
            reader.accumulate()
        except Exception as err:
            progress_dlg.done()
            properties_dlg.close()
            raise err
        properties_dlg.close()
        progress_dlg.done()
        major_name = "%s_%s" % (self.__case.get_animal_name(), self.__case['short_name'])
        imaging_signal = ImagingSignal(reader, major_name)
        imaging_signal.get_features()["ROI"] = roi_name
        SignalViewerDlg(self, imaging_signal, True).ShowModal()
        print(imaging_signal)

    def __trace_analysis_manual(self, train, properties_dlg):
        reader, isoline, sync = properties_dlg.create_reader()
        roi_name = properties_dlg.get_roi_name()
        properties_dlg.close()

        progress_dlg = ReadingProgressDialog(self, "Trace analysis", 1000, "Reading traces")
        reader.progress_bar = progress_dlg
        try:
            reader.read()
        except Exception as err:
            progress_dlg.Destroy()
            raise err
        if not reader.has_cleaned:
            print("PY Trace reading cancelled or error occured")
            del train
            del reader
            del isoline
            del sync
            progress_dlg.Destroy()
            return
        print("PY Finish of traces reading")
        progress_dlg.done()

        trace_processor = TraceProcessor(reader, isoline, sync, train, False, "mean")
        del train
        del reader
        del isoline
        del sync
        traces_dlg = TracesDlg(self, trace_processor)
        if traces_dlg.ShowModal() == wx.ID_CANCEL:
            traces_dlg.close()
            return
        traces_dlg.set_average_method_and_strategy(trace_processor)
        traces_dlg.close()

        traces = trace_processor.create_traces(self.__case, self.__case_name)
        del trace_processor
        traces.set_roi_name(roi_name)
        final_traces_dlg = FinalTracesDlg(self, traces)
        if final_traces_dlg.ShowModal() == wx.ID_CANCEL:
            final_traces_dlg.close()
            return
        save_dialog = wx.ProgressDialog("Trace analysis", "Saving the data", 100, self)
        save_dialog.Update(0)
        try:
            npz_file = final_traces_dlg.save_files(self.__case['pathname'])
            if npz_file is not None:
                self.__case['traces'].append(traces)
                traces.clean()
            else:
                del traces
            final_traces_dlg.close()
            save_dialog.Destroy()
        except Exception as err:
            save_dialog.Destroy()
            raise err
