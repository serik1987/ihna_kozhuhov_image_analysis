# -*- coding: utf-8


class PandasColumn:

    __name = None

    def __init__(self):
        pass

    def get_name(self):
        return self.__name

    def set_name(self, value):
        if isinstance(value, str) and value != "":
            self.__name = value
        else:
            raise ValueError("Incorrect column name")

    def __str__(self):
        return "Column name: %s\n" % self.get_name()

    def get_values(self, animal_list):
        raise NotImplementedError("PandasColumn.get_values")
