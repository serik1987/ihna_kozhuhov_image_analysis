# -*- coding: utf-8

from .pandascolumn import PandasColumn


class PandasFieldColumn(PandasColumn):

    __field_class = None
    __field_key = None

    def get_field_class(self):
        return self.__field_class

    def get_field_key(self):
        return self.__field_key

    def set_field_class(self, value):
        if isinstance(value, type):
            self.__field_class = value
        else:
            raise ValueError("Incorrect field class given")

    def set_field_key(self, value):
        if isinstance(value, str) and value != "":
            self.__field_key = value
        else:
            raise ValueError("Incorrect field name given")

    def __str__(self):
        S = super().__str__()
        S += "Field class: %s\n" % self.get_field_class().__name__
        S += "Field key: %s\n" % self.get_field_key()
        return S

    def get_values(self, animal_list):
        values = []
        for case in animal_list.get_animal_filter():
            if isinstance(case, self.get_field_class()):
                considering_object = case
            else:
                animal_name = case.get_animal_name()
                considering_object = animal_list[animal_name]
            values.append(considering_object[self.get_field_key()])
        return values
