# -*- coding: utf-8

from time import sleep
import wx
from wx.lib.scrolledpanel import ScrolledPanel


class AutoprocessEditor(wx.BoxSizer):

    __case = None
    __case_status_box = None
    __parent = None

    def __init__(self, parent, case):
        super().__init__(wx.HORIZONTAL)
        self.__case = case
        self.__parent = parent
        case_name = "%s_%s" % (case.get_animal_name(), case["short_name"])

        status_caption = wx.StaticText(parent, label=case_name)
        size = status_caption.GetSize()
        size.SetWidth(159)
        status_caption.SetSizeHints(size)
        self.Add(status_caption, 0, wx.RIGHT, 5)

        self.__case_status_box = wx.StaticText(parent, label="Ready",
                                               style=wx.ALIGN_LEFT | wx.ST_NO_AUTORESIZE | wx.ST_ELLIPSIZE_END)
        self.__case_status_box.SetForegroundColour("orange")
        font = self.__case_status_box.GetFont()
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self.__case_status_box.SetFont(font)
        size = self.__case_status_box.GetSize()
        self.__case_status_box.SetSizeHints(300, size[1])
        self.Add(self.__case_status_box, 0)

    def done(self):
        self.__case_status_box.SetLabel("Done")
        self.__case_status_box.SetForegroundColour("green")
        self.__parent.Update()
        self.__parent.Refresh()

    def error(self, err):
        self.__case_status_box.SetLabel(str(err))
        self.__case_status_box.SetForegroundColour("red")
        self.__parent.Update()
        self.__parent.Refresh()

    def progress_function(self, completed, total, message):
        self.__case_status_box.SetLabel("{0} ({1:.2f}% completed)".format(message, 100 * completed / total))
        self.__parent.Refresh()
        self.__parent.Update()
        print("{0} ({1}% completed)".format(message, 100 * completed / total))
        return True


class AutoprocessDlg(wx.Dialog):

    _animal_filter = None
    _editor_list = None
    _sub_dlg = None
    _parent = None

    __do_button = None
    __cancel_button = None
    __close_button = None

    __in_progress = False
    __buttons_box = None
    __panel = None
    __ready = False

    def __init__(self, parent, animal_filter, title):
        super().__init__(parent, title=title, size=(500, 800))
        self._animal_filter = animal_filter
        panel = wx.Panel(self)
        general_layout = wx.BoxSizer(wx.VERTICAL)
        main_layout = wx.BoxSizer(wx.VERTICAL)

        editor_box = ScrolledPanel(panel)
        editor_box_sizer = wx.BoxSizer(wx.VERTICAL)
        self.__editor_list = []
        for case in animal_filter:
            case_box = AutoprocessEditor(editor_box, case)
            editor_box_sizer.Add(case_box, 0, wx.BOTTOM | wx.EXPAND, 5)
            self.__editor_list.append((case, case_box))
        editor_box.SetSizer(editor_box_sizer)
        editor_box.SetupScrolling(False, True)
        main_layout.Add(editor_box, 1, wx.BOTTOM | wx.EXPAND, 10)

        buttons_box = wx.BoxSizer(wx.HORIZONTAL)
        self.__do_button = wx.Button(panel, label="Continue")
        self.__do_button.Bind(wx.EVT_BUTTON, lambda event: self.__do())
        buttons_box.Add(self.__do_button, 0, wx.RIGHT, 5)

        self.__close_button = wx.Button(panel, label="Close")
        self.__close_button.Bind(wx.EVT_BUTTON, lambda event: self.Close())
        buttons_box.Add(self.__close_button, 0)
        self.__close_button.Hide()

        main_layout.Add(buttons_box, 0, wx.ALIGN_CENTER)
        general_layout.Add(main_layout, 1, wx.ALL | wx.EXPAND, 10)
        panel.SetSizer(general_layout)
        self.Centre()

        self.__buttons_box = buttons_box
        self.__panel = panel
        self._parent = parent
        panel.Layout()

        self.__ready = self._open_sub_dlg()

    def _open_sub_dlg(self):
        return True

    def __do(self):
        self.__do_button.Hide()
        self.__panel.Layout()
        self.Layout()
        self.Update()
        self.Refresh()
        for case, case_box in self.__editor_list:
            try:
                case_full_name = "Processing case: %s_%s" % (case.get_animal_name(), case['short_name'])
                case_box.progress_function(0, 100, "In progress")
                print(case_full_name)
                self._process_single_case(case, case_box)
                case_box.done()
            except Exception as err:
                case_box.error(err)
                print("Exception name:", err.__class__.__name__)
                print("Exception:", err)
        self.__close_button.Show(True)
        self.__panel.Layout()
        self.Layout()

    def get_ready(self):
        return self.__ready

    def _process_single_case(self, case, case_box):
        raise NotImplementedError("_process_single_case")
