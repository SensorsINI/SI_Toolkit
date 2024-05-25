"""
FunctionalDict class takes a directory where the values are functions (callables)
and returns an object which resemble a dictionary which under keys return the current values of the functions.
The functions cannot take external arguments.
The main point of this structure was to access the current attributes of a class without having to update the dictionary.

HistoryClass is a directory with each value being a list together with function for updating this list.


Example:

class MyClass:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def update_parameters(self):
        self.a += 1
        self.b += 2
        self.c += 3

my_class = MyClass(1, 2, 3)

my_class_values = {
            'a': lambda: my_class.a,
            'b': lambda: my_class.b,
            'c': lambda: my_class.c
        }

my_class_values_dictified = FunctionalDict(my_class_values)

history_class = HistoryClass()
history_class.add_keys(current_values.keys())
my_class.update_parameters()
history_class.update_history(current_values)
my_class.update_parameters()
history_class.update_history(current_values)
"""

class FunctionalDict:
    def __init__(self, dict):
        self._values = dict

    def __getitem__(self, key):
        if key in self._values:
            return self._values[key]()
        raise KeyError(f"Key {key} not found in CurrentValues.")

    def __setitem__(self, key, value):
        if key in self._values and callable(value):
            self._values[key] = value
        else:
            raise ValueError("Value must be a callable returning the value for the key.")

    def __delitem__(self, key):
        if key in self._values:
            del self._values[key]
        else:
            raise KeyError(f"Key {key} not found in CurrentValues.")

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def keys(self):
        return self._values.keys()

    def values(self):
        return [func() for func in self._values.values()]

    def items(self):
        return [(key, func()) for key, func in self._values.items()]

    def __repr__(self):
        return f"{self.__class__.__name__}({{ {', '.join(f'{k}: {v()}' for k, v in self._values.items())} }})"


class HistoryClass:
    def __init__(self):
        self.history = {}

    def update_history(self, current_values):
        for key in current_values.keys():
            if key in self.history.keys():
                self.history[key].append(current_values[key])

    def history_reset(self):
        self.history = {key: [] for key in self.history.keys()}

    def add_key(self, key):
        if self.is_history_empty():
            if key not in self.history.keys():
                self.history[key] = []

    def remove_key(self, key):
        if self.is_history_empty():
            if key in self.history.keys():
                del self.history[key]

    def add_keys(self, keys):
        for key in keys:
            self.add_key(key)

    def remove_keys(self, keys):
        for key in keys:
            self.remove_key(key)

    def is_history_empty(self):
        return all(len(entries) == 0 for entries in self.history.values())

    def keys(self):
        return self.history.keys()

    def values(self):
        return self.history.values()

    def items(self):
        return self.history.items()

    def __getitem__(self, key):
        return self.history[key]

    def __setitem__(self, key, value):
        if key in self.history.keys():
            self.history[key] = value
        else:
            raise KeyError(f"Key '{key}' is not defined in the history keys.")

    def __delitem__(self, key):
        if key in self.history.keys():
            del self.history[key]
        else:
            raise KeyError(f"Key '{key}' is not defined in the history keys.")

    def __iter__(self):
        return iter(self.history)

    def __repr__(self):
        return f"{self.__class__.__name__}({{ {', '.join(f'{k}: {v}' for k, v in self.history.items())} }})"