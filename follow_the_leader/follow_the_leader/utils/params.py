#!/usr/bin/env python3
import json
import numpy as np
from copy import deepcopy

class Params:
    """A json serializable class for storing information"""
    def __init__(
        self,
        json_string = "{}",
    ):
        self.load_json(json_string)

    def is_param(self, str):
        return str in self.__dict__.keys()

    def get_param(self, str):
        """ check if a parameter exists in the dictionary and return it if it does, None on failure

        :param str: parameter name
        :return: parameter value or None
        """
        return self.__dict__.get(str)
    
    def add_param(self, str, val):
        self.__dict__[str] = val

    def update_param(self, str, val):
        self.__dict__[str] = val

    def json_dump(self):
        return json.dumps(self.__dict__, indent=4)

    def load_json(self, json_str: str):
        self.__dict__ = json.loads(json_str)
        for key in self.__dict__.keys():
            if isinstance(self.__dict__[key], list):
                self.__dict__[key] = np.array(self.__dict__[key])
    
    def __repr__(self):
        return f'{self.json_dump()}'

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result


