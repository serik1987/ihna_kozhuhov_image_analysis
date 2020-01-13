# -*- coding: utf-8

from .isoline import IsolineEditor
from .noisoline import NoIsolineEditor
from .linearfitisoline import LinearFitIsolineEditor
from .timeaverageisoline import TimeAverageIsolineEditor

editor_list = [NoIsolineEditor, LinearFitIsolineEditor, TimeAverageIsolineEditor]
