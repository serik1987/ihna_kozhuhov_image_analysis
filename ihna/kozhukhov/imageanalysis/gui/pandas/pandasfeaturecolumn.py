# -*- coding: utf-8

import re
from .pandascolumn import PandasColumn


class PandasFeatureColumn(PandasColumn):

    __prefix_name = None
    __postfix_name = None
    __minor_name = None
    __feature_name = None

    def get_prefix_name(self):
        return self.__prefix_name

    def get_postfix_name(self):
        return self.__postfix_name

    def get_minor_name(self):
        return self.__minor_name

    def get_feature_name(self):
        return self.__feature_name

    def set_prefix_name(self, value):
        self.__prefix_name = value

    def set_postfix_name(self, value):
        self.__postfix_name = value

    def set_minor_name(self, value):
        self.__minor_name = value

    def set_feature_name(self, value):
        self.__feature_name = value

    def clear_prefix_name(self):
        self.__prefix_name = None

    def clear_postfix_name(self):
        self.__postfix_name = None

    def __str__(self):
        S = super().__str__()
        if self.get_prefix_name() is None:
            S += "Prefix name is not specified\n"
        else:
            S += "Prefix name: %s\n" % self.get_prefix_name()
        if self.get_postfix_name() is None:
            S += "Postfix name is not specified\n"
        else:
            S += "Postfix name: %s\n" % self.get_postfix_name()
        S += "Minor name: %s\n" % self.get_minor_name()
        S += "Feature name: %s\n" % self.get_feature_name()
        return S

    def get_values(self, animal_list):
        values = []
        for case in animal_list.get_animal_filter():
            try:
                value = None
                for data in case.data():
                    major_name = data.get_features()['major_name']
                    minor_name = data.get_features()['minor_name']
                    pos = major_name.find('_')
                    major_name = major_name[pos+1:]
                    pos = major_name.find(case['short_name'])
                    prefix_name = major_name[:pos]
                    postfix_name = major_name[pos + len(case['short_name']):]
                    condition1 = self.get_prefix_name() is None or self.get_prefix_name() == prefix_name
                    condition2 = self.get_postfix_name() is None or self.get_postfix_name() == postfix_name
                    condition3 = minor_name == self.get_minor_name()
                    if condition1 and condition2 and condition3:
                        feature_name = self.get_feature_name()
                        try:
                            value = data.get_features()[feature_name]
                            try:
                                value = float(value)
                            except ValueError:
                                pass
                        except KeyError:
                            raise ValueError("Feature not found")
                    if value is not None:
                        break
                if value is None:
                    raise ValueError("The data is not present in a given case")
                values.append(value)
            except Exception as err:
                values.append(str(err))
        return values
