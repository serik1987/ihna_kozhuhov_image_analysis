# -*- coding: utf-8

from scipy.signal import iirnotch
from .filterbox import FilterBox


class IirNotchBox(FilterBox):

    _filter_properties = ["narrowband"]

    def _get_filter_name(self):
        return "iirnotch"

    def get_coefficients(self):
        F0 = self.get_center_frequency()
        dF = self.get_bandwidth()
        Fmax = 0.5 * self._sample_rate
        w0 = F0 / Fmax
        Q = F0 / dF
        b, a = iirnotch(w0, Q)
        return b, a
