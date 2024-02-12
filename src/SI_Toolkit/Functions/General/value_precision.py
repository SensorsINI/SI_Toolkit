# This is equivalent to the functions we used in C, up to the wrapping of negative numbers
from SI_Toolkit.computation_library import NumpyLibrary


def float_to_ap_fixed(value, total_bits=20, integer_bits=6, rounding=True, lib=NumpyLibrary):
    # Calculate the scale for the fractional part
    scale = 2 ** (total_bits - integer_bits)

    # Convert the float to the fixed-point representation scale
    scaled_value = value * scale

    # Round or truncate using NumPy functions for vectorized operations
    if rounding:
        fixed_point_value = lib.cast(lib.rint(scaled_value), lib.int32)
    else:
        fixed_point_value = lib.cast(lib.floor(scaled_value), lib.int32)  # Using floor for truncation

    # Maximum and minimum representable values calculation
    max_val = 2 ** (total_bits - 1) - 1
    min_val = -2 ** (total_bits - 1)

    # Implement wrapping
    # NumPy's modulo operation is vectorized, so it works element-wise on arrays
    fixed_point_value = (fixed_point_value - min_val) % (max_val - min_val + 1) + min_val

    # Convert back to float to simulate fixed-point precision
    return lib.cast(fixed_point_value / scale, lib.float32)



def set_value_precision(value, precision, lib=NumpyLibrary):
    if precision == 'float':
        pass
    elif precision[:len('ap_fixed')] == 'ap_fixed':
        precision = precision[len('ap_fixed<'):-1]
        precision = precision.split(',')
        total_bits = int(precision[0])
        integer_bits = int(precision[1])
        value = float_to_ap_fixed(value, total_bits=total_bits, integer_bits=integer_bits, rounding=True, lib=lib)
    return value
