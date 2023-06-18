from dataclasses import dataclass
from typing import Any, Dict
import json


def dict_schema_repr(obj: Dict[Any, Any]) -> Dict[str, Dict | str]:
    class _Unknown:
        def __repr__(self):
            return 'Unknown'
    Unknown = _Unknown()

    @dataclass
    class List:
        value: Any

        def __repr__(self):
            if type(self.value) == dict:
                return f'List[{json.dumps(self.value, sort_keys=True, indent=4)})'
            return f'List[{repr(self.value)}]'

    @dataclass
    class Type:
        value: Any

        def __repr__(self):
            return f'Type[{repr(self.value)}]'


    result = {}
    for key, value in obj.items():
        new_value = None
        if type(value) == dict:
            new_value = dict_schema_repr(value)
        elif type(value) == list:
            if len(value) > 0:
                list_type = type(value[0])
                if list_type is dict:
                    list_type = dict_schema_repr(value[0])
                new_value = Type(list_type)
            else:
                new_value = List(Unknown)
        else:
            if value == None:
                new_value = Type(Unknown)
            else:
                new_value = Type(type(value))
        result[key] = new_value
    return result
