# -*- coding: utf-8

from scipy.signal import iirpeak
from .filterbox import FilterBox


class IirPeakBox(FilterBox):

    _filter_properties = ["narrowband"]

    def _get_filter_name(self):
        return "iirpeak"

    def get_coefficients(self):
        f0 = self.get_center_frequency()
        dF = self.get_bandwidth()
        Fmax = 0.5 * self._sample_rate
        w0 = f0 / Fmax
        Q = f0 / dF
        b, a = iirpeak(w0, Q)
        return b, a
