# -*- coding: utf-8

from .filter import Filter


class AnimalFilter(Filter):
    """
    Returns an object that allows set filter on animals and iterate throw animals
    """

    __animal_index = None
    __case_filter = None

    def get_filter_fields(self):
        """
        Returns the list containing names of all fields available for the filtration
        """
        return ["specimen", "conditions", "recording_site"]

    def __iter__(self):
        return self

    def __next__(self):
        from ihna.kozhukhov.imageanalysis.manifest import CasesList
        next_case = None

        while next_case is None:
            animal_keys = self.get_data().get_animal_keys()
            if self.__case_filter is None:
                if self.__animal_index is None:
                    self.__animal_index = 0
                else:
                    self.__animal_index += 1
                if self.__animal_index >= len(animal_keys):
                    self.__animal_index = None
                    raise StopIteration()
                animal_key = animal_keys[self.__animal_index]
                animal = self.get_data()[animal_key]
                include = True
                for field_name in self.get_filter_fields():
                    field_value = animal[field_name]
                    template = self[field_name]
                    if field_value.find(template) == -1:
                        include = False
                if include:
                    case_list = CasesList(animal)
                    self.__case_filter = case_list.get_case_filter()
                continue
            try:
                next_case = self.__case_filter.__next__()
            except StopIteration:
                self.__case_filter = None
                next_case = None

        return next_case

    def reset_iteration(self):
        self.__animal_index = None
        if self.__case_filter is not None:
            self.__case_filter.reset_iteration()
        self.__case_filter = None
