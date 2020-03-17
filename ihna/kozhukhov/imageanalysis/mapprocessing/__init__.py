# -*- coding: utf-8

import numpy as np
from ihna.kozhukhov.imageanalysis import ImagingMap
from ihna.kozhukhov.imageanalysis._imageanalysis import convolve as _convolve


def spatial_filter(complex_map: ImagingMap, dradius=0, dradiusbig=0):
    """
    Provides spatial filter of the map

    Arguments:
        complex_map - the complex map to filter
        dradius - radius of the spatial high-pass filter or 0 is spatial high-pass filtration shall not be applied
        dradiusbig - radius of the spatial low-pass filter or 0 is spatial low-pass filtration shall not be applied

    Returns:
        An instance of the ImagingMap which in turn represents a complex map
    """
    data = complex_map.get_data()
    if data.dtype != np.complex:
        raise ValueError("This processor works with maps expressed in complex numbers only")
    if dradius > 0:
        result_real = _convolve(data.real, dradius)
        result_imag = _convolve(data.imag, dradius)
    else:
        result_real = data.real
        result_imag = data.imag

    if dradiusbig > 0:
        background_real = _convolve(result_real, dradiusbig)
        background_imag = _convolve(result_imag, dradiusbig)
        result_real -= background_real
        result_imag -= background_imag

    features = complex_map.get_features().copy()
    features['original_map'] = complex_map.get_full_name()
    features['minor_name'] = 'filt'
    if dradius > 0:
        features['HPF_radius'] = dradius
    if dradiusbig > 0:
        features['LPF_radius'] = dradiusbig
    result_data = result_real + 1j * result_imag
    result_map = ImagingMap(features, result_data)

    return result_map
