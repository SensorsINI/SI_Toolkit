import numpy as np

from SI_Toolkit.computation_library import TensorType


class VariableParameters:
    def __init__(self, lib):
        self.lib = lib
        self.set_attributes = self._set_attributes


    def set_attributes(self, initial_environment_attributes, device=None):
        if device is not None:
            self.set_attributes = self.lib.set_device(device)(self._set_attributes)
        self.set_attributes(initial_environment_attributes)

    def _set_attributes(self, initial_environment_attributes, **kwargs):
        # Set properties like target positions on this controller
        for p, v in initial_environment_attributes.items():
            if type(v) in {np.ndarray, float, int, bool}:
                data_type = getattr(v, "dtype", self.lib.float32)
                data_type = self.lib.int32 if data_type == int else self.lib.float32
                v = self.lib.to_variable(v, data_type)
            setattr(self, p, v)

    def update_attributes(self, updated_attributes: "dict[str, TensorType]"):
        for attribute, new_value in updated_attributes.items():
            attr = getattr(self, attribute)
            self.lib.assign(attr, self.lib.to_tensor(new_value, attr.dtype))
