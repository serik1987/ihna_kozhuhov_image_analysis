# -*- coding: utf-8

from .filter import Filter


class CaseFilter(Filter):
    """
    Returns object that allows to filter on cases and iterate throw cases
    """

    __index = None

    def get_filter_fields(self):
        """
        Returns the list containing names of all fields available for the filtration
        """
        return ["short_name", "long_name", "stimulation", "additional_stimulation", "special_conditions",
                "additional_information"]

    def __iter__(self):
        return self

    def __next__(self):
        next_case = None
        all_cases = self.get_data().get_all_cases()

        while next_case is None:
            if self.__index is None:
                self.__index = 0
            else:
                self.__index += 1
            if self.__index >= len(all_cases):
                self.__index = None
                raise StopIteration()
            _next_case = all_cases[self.__index]
            if _next_case['auto']:
                include = True
                for field_name in self.get_filter_fields():
                    field_value = _next_case[field_name]
                    template = self[field_name]
                    if field_value.find(template) == -1:
                        include = False
                if include:
                    next_case = _next_case

        return next_case
