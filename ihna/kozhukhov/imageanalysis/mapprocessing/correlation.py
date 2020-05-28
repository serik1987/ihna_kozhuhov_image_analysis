# -*- coding: utf-8

import numpy as np
from astropy import units as u
from astropy.stats import circcorrcoef
from ihna.kozhukhov.imageanalysis import ImagingMap


def ordinary_correlation(map1: ImagingMap, map2: ImagingMap):
    """
    Computes ordinary correlation between two maps. This correlation is suitable for non-periodic
    maps (e.g., amplitude or oscillatory maps)

    Arguments:
        map1, map2 - maps between which correlation shall be computed (an instance of
        ihna.kozhukhov.imageanalysis.ImagingMap)
    """
    first_map = map1.get_data()
    second_map = map2.get_data()
    first_map = np.reshape(first_map, first_map.size)
    second_map = np.reshape(second_map, second_map.size)
    R = np.corrcoef(first_map, second_map)
    r = R[0][1]
    return r


def circular_correlation(map1: ImagingMap, map2: ImagingMap):
    """
    Computes circular correlation between two maps. Use this method for any maps that represent circular
    values (e.g., orientation or direction maps)
    """
    first_map = map1.get_data()
    second_map = map2.get_data()
    first_map = np.reshape(first_map, first_map.size)
    second_map = np.reshape(second_map, second_map.size)
    h = map1.get_harmonic()
    alpha = first_map * h * u.rad
    beta = second_map * h * u.rad
    r = circcorrcoef(alpha, beta)
    value = float(r)
    return value
