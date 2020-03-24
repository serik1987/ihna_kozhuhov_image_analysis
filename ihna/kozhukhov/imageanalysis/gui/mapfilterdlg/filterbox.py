# -*- coding: utf-8

import wx


class FilterBox(wx.BoxSizer):

    _dlg = None
    _parent = None

    _filter_properties = None
    _filter_modes = ["low-pass", "high-pass", "band-pass", "notch"]
    __btypes = {
        "low-pass": "lowpass",
        "high-pass": "highpass",
        "band-pass": "bandpass",
        "notch": "bandstop"
    }
    __default_transition_band = None
    __default_bandpass_loss = None
    __default_attenuation = None
    __default_bandpass_ripples = None
    __default_min_attenuation = None
    __default_order = None
    _sample_rate = None

    __filter_widgets = None
    __filter_type_box = None
    __filter_lf_caption = None
    __filter_lf_box = None
    __filter_hf_caption = None
    __filter_hf_box = None
    __transition_band_box = None
    __bandpass_loss_box = None
    __attenuation_box = None
    __rippable_box = None
    __min_attenuation_box = None
    __order_box = None
    __center_frequency_box = None
    __bandwidth = None

    __order = None
    __wn = None

    def __init__(self, dlg, parent, sample_rate, default_transition_band=0.05, default_bandpass_loss=0.04,
                 default_attenuation=50, default_bandpass_ripples=0.04, default_min_attenuation=50,
                 default_order=4):
        super().__init__(wx.VERTICAL)
        self.__filter_widgets = []
        self._dlg = dlg
        self._parent = parent
        self.__default_transition_band = default_transition_band
        self.__default_bandpass_loss = default_bandpass_loss
        self.__default_attenuation = default_attenuation
        self.__default_bandpass_ripples = default_bandpass_ripples
        self.__default_min_attenuation = default_min_attenuation
        self.__default_order = default_order
        self._sample_rate = sample_rate
        self._create_widgets()

    def _get_filter_name(self):
        raise NotImplementedError("_get_filter_name")

    def _create_widgets(self):
        if "broadband" in self._filter_properties:
            filter_type_layout = wx.BoxSizer(wx.HORIZONTAL)
            filter_type_caption = wx.StaticText(self._parent, label="Filter subtype")
            filter_type_layout.Add(filter_type_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
            self.__filter_widgets.append(filter_type_caption)

            self.__filter_type_box = wx.Choice(self._parent,
                                               choices=["Low-pass filter", "High-pass filter",
                                                        "Band-pass filter", "Notch filter"])
            self.__filter_type_box.SetSelection(0)
            filter_type_layout.Add(self.__filter_type_box, 1, wx.EXPAND)
            self.__filter_widgets.append(self.__filter_type_box)
            self.Add(filter_type_layout, 0, wx.EXPAND | wx.BOTTOM, 5)
            self.__filter_type_box.Bind(wx.EVT_CHOICE, lambda event: self.__select_filter_mode())

            filter_lf_layout = wx.BoxSizer(wx.HORIZONTAL)
            self.__filter_lf_caption = wx.StaticText(self._parent, label="Low frequency, Hz")
            filter_lf_layout.Add(self.__filter_lf_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
            self.__filter_widgets.append(self.__filter_lf_caption)

            self.__filter_lf_box = wx.TextCtrl(self._parent)
            filter_lf_layout.Add(self.__filter_lf_box, 1, wx.EXPAND)
            self.__filter_widgets.append(self.__filter_lf_box)
            self.Add(filter_lf_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

            filter_hf_layout = wx.BoxSizer(wx.HORIZONTAL)
            self.__filter_hf_caption = wx.StaticText(self._parent, label="High frequency, Hz")
            filter_hf_layout.Add(self.__filter_hf_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
            self.__filter_widgets.append(self.__filter_hf_caption)

            self.__filter_hf_box = wx.TextCtrl(self._parent)
            filter_hf_layout.Add(self.__filter_hf_box, 1, wx.EXPAND)
            self.__filter_widgets.append(self.__filter_hf_box)
            self.Add(filter_hf_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

            self.__select_filter_mode()

        if "standard" in self._filter_properties:
            transition_band_layout = wx.BoxSizer(wx.HORIZONTAL)
            transition_band_caption = wx.StaticText(self._parent, label="Transition band, Hz")
            transition_band_layout.Add(transition_band_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
            self.__filter_widgets.append(transition_band_caption)

            self.__transition_band_box = wx.TextCtrl(self._parent, value=str(self.__default_transition_band))
            transition_band_layout.Add(self.__transition_band_box, 1, wx.EXPAND)
            self.__filter_widgets.append(self.__transition_band_box)
            self.Add(transition_band_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

            bandpass_loss_layout = wx.BoxSizer(wx.HORIZONTAL)
            bandpass_loss_caption = wx.StaticText(self._parent, label="Maximum loss in the passband, dB")
            bandpass_loss_layout.Add(bandpass_loss_caption, 0, wx.RIGHT | wx.ALIGN_CENTER_VERTICAL, 5)
            self.__filter_widgets.append(bandpass_loss_caption)

            self.__bandpass_loss_box = wx.TextCtrl(self._parent, value=str(self.__default_bandpass_loss))
            bandpass_loss_layout.Add(self.__bandpass_loss_box, 1, wx.EXPAND)
            self.__filter_widgets.append(self.__bandpass_loss_box)
            self.Add(bandpass_loss_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

            attenuation_layout = wx.BoxSizer(wx.HORIZONTAL)
            attenuation_caption = wx.StaticText(self._parent, label="Minimum attenuation in the stopband, dB")
            attenuation_layout.Add(attenuation_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
            self.__filter_widgets.append(attenuation_caption)

            self.__attenuation_box = wx.TextCtrl(self._parent, value=str(self.__default_attenuation))
            attenuation_layout.Add(self.__attenuation_box, 1, wx.EXPAND)
            self.__filter_widgets.append(self.__attenuation_box)
            self.Add(attenuation_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        if "rippable" in self._filter_properties:
            rippable_layout = wx.BoxSizer(wx.HORIZONTAL)
            rippable_caption = wx.StaticText(self._parent, label="Bandpass ripples, dB")
            rippable_layout.Add(rippable_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
            self.__filter_widgets.append(rippable_caption)

            self.__rippable_box = wx.TextCtrl(self._parent, value=str(self.__default_bandpass_ripples))
            rippable_layout.Add(self.__rippable_box, 1, wx.EXPAND)
            self.__filter_widgets.append(self.__rippable_box)
            self.Add(rippable_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        if "self_attenuatable" in self._filter_properties:
            min_attenuation_layout = wx.BoxSizer(wx.HORIZONTAL)
            min_attenuation_caption = wx.StaticText(self._parent, label="Min. attenuation, dB")
            min_attenuation_layout.Add(min_attenuation_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)
            self.__filter_widgets.append(min_attenuation_caption)

            self.__min_attenuation_box = wx.TextCtrl(self._parent, value=str(self.__default_min_attenuation))
            min_attenuation_layout.Add(self.__min_attenuation_box, 1, wx.EXPAND)
            self.__filter_widgets.append(self.__min_attenuation_box)
            self.Add(min_attenuation_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        if "manual" in self._filter_properties:
            order_layout = wx.BoxSizer(wx.HORIZONTAL)
            order_caption = wx.StaticText(self._parent, label="Order")
            self.__filter_widgets.append(order_caption)
            order_layout.Add(order_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

            self.__order_box = wx.TextCtrl(self._parent, value=str(self.__default_order))
            self.__filter_widgets.append(self.__order_box)
            order_layout.Add(self.__order_box, 1, wx.EXPAND)
            self.Add(order_layout, 0, wx.EXPAND | wx.BOTTOM, 5)

        if "narrowband" in self._filter_properties:
            center_frequency_layout = wx.BoxSizer(wx.HORIZONTAL)
            center_frequency_caption = wx.StaticText(self._parent, label="Center frequency, Hz")
            self.__filter_widgets.append(center_frequency_caption)
            center_frequency_layout.Add(center_frequency_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

            self.__center_frequency_box = wx.TextCtrl(self._parent)
            self.__filter_widgets.append(self.__center_frequency_box)
            center_frequency_layout.Add(self.__center_frequency_box, 1, wx.EXPAND)
            self.Add(center_frequency_layout, 0, wx.BOTTOM | wx.EXPAND, 5)

            bandwidth_layout = wx.BoxSizer(wx.HORIZONTAL)
            bandwidth_caption = wx.StaticText(self._parent, label="Frequency bandwidth, Hz")
            self.__filter_widgets.append(bandwidth_caption)
            bandwidth_layout.Add(bandwidth_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.RIGHT, 5)

            self.__bandwidth_box = wx.TextCtrl(self._parent)
            self.__filter_widgets.append(self.__bandwidth_box)
            bandwidth_layout.Add(self.__bandwidth_box, 1, wx.EXPAND)
            self.Add(bandwidth_layout, 0, wx.BOTTOM | wx.EXPAND, 5)

        filter_rate_layout = wx.BoxSizer(wx.HORIZONTAL)
        filter_rate_caption = wx.StaticText(self._parent, label="Your rate (1-10)")
        filter_rate_layout.Add(filter_rate_caption, 0, wx.ALIGN_CENTER_VERTICAL | wx.BOTTOM, 5)
        self.__filter_widgets.append(filter_rate_caption)

        filter_rate_box = wx.TextCtrl(self._parent)
        filter_rate_layout.Add(filter_rate_box, 1, wx.EXPAND)
        self.__filter_widgets.append(filter_rate_box)
        self.Add(filter_rate_layout, 0, wx.BOTTOM | wx.EXPAND, 5)

    def show(self):
        for widget in self.__filter_widgets:
            widget.Show(True)
        # self.Layout()

    def hide(self):
        for widget in self.__filter_widgets:
            widget.Show(False)
        # self.Layout()

    def get_filter_mode(self):
        if "broadband" in self._filter_properties:
            mode_index = self.__filter_type_box.GetSelection()
            mode = self._filter_modes[mode_index]
            return mode
        else:
            raise AttributeError("This filter is not broadband")

    def get_low_frequency(self):
        if self.get_filter_mode() == "low-pass":
            raise AttributeError("The parameter is not applicable for low-pass filter")
        else:
            try:
                value = float(self.__filter_lf_box.GetValue())
                if value <= 0:
                    raise RuntimeError("LF value is negative or zero")
                if value >= 0.5 * self._sample_rate:
                    raise RuntimeError("LF value is too high")
                return value
            except ValueError:
                raise ValueError("Please, enter a correct value for low frequency")

    def get_high_frequency(self):
        if self.get_filter_mode() == "high-pass":
            raise AttributeError("The parameter is not applicable for high-pass filter")
        else:
            try:
                value = float(self.__filter_hf_box.GetValue())
                if value <= 0:
                    raise RuntimeError("HF value is negative or zero")
                if value >= 0.5 * self._sample_rate:
                    raise RuntimeError("HF value is too high")
                return value
            except ValueError:
                raise ValueError("Please, enter a correct value for high frequency")

    def get_transition_band(self):
        if "standard" in self._filter_properties:
            try:
                value = float(self.__transition_band_box.GetValue())
                if value > 0:
                    return value
                else:
                    raise RuntimeError("Transition band shall be positive")
            except ValueError:
                raise ValueError("Please, enter a correct value of the transition band")
        else:
            raise AttributeError("The transition band is not available for this type of filter")

    def get_bandpass_loss(self):
        if "standard" in self._filter_properties:
            try:
                value = float(self.__bandpass_loss_box.GetValue())
                if value > 0:
                    return value
                else:
                    raise RuntimeError("Bandpass loss shall be positive value expressed in Hz")
            except ValueError:
                raise ValueError("Please, enter a correct value of the bandpass")
        else:
            raise AttributeError("The bandpass loss is not available for this type of filter")

    def get_attenuation(self):
        if "standard" in self._filter_properties:
            try:
                value = float(self.__attenuation_box.GetValue())
                if value > 0:
                    return value
                else:
                    raise RuntimeError("The attenuation shall be positive value expressed in dB")
            except ValueError:
                raise ValueError("Please, enter a correct value of the attenuation")
        else:
            raise AttributeError("The attenuation box is not available for this type of filter")

    def get_ripples(self):
        if "rippable" in self._filter_properties:
            try:
                value = float(self.__rippable_box.GetValue())
                if value > 0:
                    return value
                else:
                    raise RuntimeError("Bandpass ripples must be positive value expressed in dB")
            except ValueError:
                raise ValueError("Please, enter a correct value of the bandpass ripples")
        else:
            raise AttributeError("Bandpass ripples can't be set for this type of filter")

    def get_min_attenuation(self):
        if "self_attenuatable" in self._filter_properties:
            try:
                value = float(self.__min_attenuation_box.GetValue())
                if value > 0:
                    return value
                else:
                    raise RuntimeError("Min attenuation must be positive value expressed in dB")
            except ValueError:
                raise ValueError("Please, enter a correct value of the minimum attenuation")
        else:
            raise AttributeError("The min attenuation is not applicable for this type of filter")

    def get_order(self):
        if "manual" in self._filter_properties:
            try:
                value = int(self.__order_box.GetValue())
                if value > 1:
                    self._set_current_order(value)
                    return value
                else:
                    raise RuntimeError("Filter order shall be at least 2")
            except ValueError:
                raise ValueError("Filter order shall be a correct integer value")
        else:
            raise AttributeError("This type of filter supports automatic order selection. Fill another values")

    def get_center_frequency(self):
        if "narrowband" in self._filter_properties:
            try:
                value = float(self.__center_frequency_box.GetValue())
                if value <= 0:
                    raise RuntimeError("The center frequency shall be strongly positive")
                if value >= 0.5 * self._sample_rate:
                    raise RuntimeError("The center frequency is too high")
                return value
            except ValueError:
                raise ValueError("Please, enter a correct value of the center frequency")
        else:
            raise AttributeError("This parameter is not applicable for the current filter")

    def get_bandwidth(self):
        if "narrowband" in self._filter_properties:
            try:
                value = float(self.__bandwidth_box.GetValue())
                if value <= 0:
                    raise RuntimeError("The frequency bandwidth shall be strongly positive")
                if value >= 0.5 * self._sample_rate:
                    raise RuntimeError("The frequency bandwidth is too high")
                return value
            except ValueError:
                raise ValueError("Please, enter a correct value of the frequency bandwidth")
        else:
            raise AttributeError("This parameter is not applicable for the current filter")

    def __select_filter_mode(self):
        mode = self.get_filter_mode()

        if mode == "low-pass":
            self.__filter_lf_caption.Enable(False)
            self.__filter_lf_box.Enable(False)
            self.__filter_lf_box.SetValue("")
        else:
            self.__filter_lf_caption.Enable(True)
            self.__filter_lf_box.Enable(True)

        if mode == "high-pass":
            self.__filter_hf_box.Enable(False)
            self.__filter_hf_box.SetValue("")
            self.__filter_hf_caption.Enable(False)
        else:
            self.__filter_hf_box.Enable(True)
            self.__filter_hf_caption.Enable(True)

    def get_coefficients(self):
        raise NotImplementedError("get_coefficients")

    def get_current_order(self):
        if self.__order is not None:
            return self.__order
        else:
            raise AttributeError("Filter order is not available")

    def _set_current_order(self, value):
        self.__order = int(value)

    def get_Wn(self):
        if self.__wn is not None:
            return self.__wn
        else:
            raise AttributeError("Filter Wn is not available")

    def _set_Wn(self, value):
        self.__wn = value

    def _get_btype(self):
        filter_mode = self.get_filter_mode()
        return self.__btypes[filter_mode]

    def _get_std_passband(self):
        filter_mode = self.get_filter_mode()
        Fmax = 0.5 * self._sample_rate
        w = None
        if filter_mode == "low-pass":
            w = self.get_high_frequency() / Fmax
        elif filter_mode == "high-pass":
            w = self.get_low_frequency() / Fmax
        else:
            w1 = self.get_low_frequency() / Fmax
            w2 = self.get_high_frequency() / Fmax
            w = [w1, w2]
        return w

    def _get_passband(self):
        filter_mode = self.get_filter_mode()
        Fmax = 0.5 * self._sample_rate
        dW = self.get_transition_band() / Fmax
        w1, w2 = None, None
        wp, ws = -1.0, -1.0

        if filter_mode != "low-pass":
            w1 = self.get_low_frequency() / Fmax
        if filter_mode != "high-pass":
            w2 = self.get_high_frequency() / Fmax

        if filter_mode == "low-pass":
            wp = w2
            ws = w2 + dW
            if ws > 1.0:
                ws = 1.0 - 1e-10

        if filter_mode == "high-pass":
            wp = w1
            ws = w1 - dW
            if ws < 0.0:
                ws = 1e-10

        if filter_mode == "band-pass":
            wp = [w1, w2]
            ws = [w1 - dW, w2 + dW]
            if ws[0] < 0.0:
                ws[0] = 1e-10
            if ws[1] > 1.0:
                ws[1] = 1.0 - 1e-10

        if filter_mode == "notch":
            wp = [w1, w2]
            ws = [w1 + dW, w2 - dW]
            if ws[0] > ws[1]:
                ws[0] = ws[1] - 1e-10

        return wp, ws

    def get_filter_description(self):
        descriptions = []
        if "manual" in self._filter_properties:
            descriptions.append("%dth order %s" % (self.get_order(), self._get_filter_name()))
        else:
            descriptions.append(self._get_filter_name())
        if "broadband" in self._filter_properties:
            if self.get_filter_mode() == "low-pass":
                descriptions.append("low-pass (HF=%1.2f Hz)" % self.get_high_frequency())
            elif self.get_filter_mode() == "high-pass":
                descriptions.append("high-pass (LF=%1.2f Hz)" % self.get_low_frequency())
            elif self.get_filter_mode() == "band-pass":
                filter_range = self.get_filter_mode(), self.get_low_frequency(), self.get_high_frequency()
                descriptions.append("%s (LF = %1.2f Hz, HF = %1.2f) Hz" % filter_range)
        if "narrowband" in self._filter_properties:
            Fc = self.get_center_frequency()
            dF = self.get_bandwidth()
            descriptions.append("narrow (Fc = %1.2f Hz, dF = %1.2f Hz)".format(Fc, dF))
        if "standard" in self._filter_properties:
            tb = self.get_transition_band()
            attenuation = self.get_attenuation()
            descriptions.append("auto order select (transition band %1.2f Hz, attenuation %1.2f Hz)"
                                % (tb, attenuation))
        if "rippable" in self._filter_properties:
            descriptions.append("band-pass ripples %1.2f dB" % self.get_ripples())
        if "self_attenuatable" in self._filter_properties:
            descriptions.append("min. attenuation %1.2f dB" % self.get_min_attenuation())
        return ", ".join(descriptions)
