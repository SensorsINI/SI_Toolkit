import numpy as np


def return_hyperbolic_function(point_1, point_2, mode=1, slope=None, fixed_point=None, return_parameters=False):
    if slope is None and fixed_point is None:
        raise ValueError('slope and fixed_point cannot be both None!')

    x1 = point_1[0]
    y1 = point_1[1]

    x2 = point_2[0]
    y2 = point_2[1]

    x_intercept = -(x2-x1)
    y_intercept = y2-y1

    if fixed_point is None:
        a = slope
        delta_b = x_intercept ** 2 + 4 * (x_intercept / y_intercept) * a
        b = (-x_intercept + mode * np.sqrt(delta_b)) / 2
        c = y_intercept - (a / b)
    else:

        fixed_point = (fixed_point[0] - x2, fixed_point[1] - y1)

        x_target, y_target = fixed_point[0], fixed_point[1]
        A = x_target*y_target/(1-(x_target/x_intercept)-(y_target/y_intercept))
        b = A/y_intercept
        c = -A/x_intercept
        a = A-b*c

    betha = b-x2
    gamma = c+y1
    def hyperbolic_function(x):
        return a/(x+betha) + gamma

    def hyperbolic_function_derivative(x):
        return a/((x+betha)**2)

    if return_parameters:
        return hyperbolic_function, hyperbolic_function_derivative, a, betha, gamma
    else:
        return hyperbolic_function, hyperbolic_function_derivative, a


def return_hyperbolic_function_false(slope, x_intercept, y_intercept, mode=1):
    a = slope  # Concaveness slope
    A = y_intercept  # y-intercept
    B = x_intercept  # x_intercet

    def hyperbolic_function(x):
        x = np.clip(x, 0.0, B)
        return a / (x + (a / A)) - a / (B + (a / A))

    return hyperbolic_function

if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import platform
    if platform.system() == 'Darwin':
        import matplotlib
        matplotlib.use('MacOSX')

    a = []

    x_intercept = -1.0
    y_intercept = 1.0

    point_1 = (0.0, 0.0)
    point_2 = (2.0, 1000.0)

    mode = 1
    fixed_point = (1.0-np.cos(0.15), 1.0)

    x_min = np.minimum(point_1[0], point_2[0])
    x_max = np.maximum(point_1[0], point_2[0])

    y_min = np.minimum(point_1[1], point_2[1])
    y_max = np.maximum(point_1[1], point_2[1])

    x = np.linspace(x_min, x_max, 100)

    y_list = []
    y_derivative_list = []
    slope_list = []
    for slope in a:
        hyperbolic_f, hyperbolic_f_derivative, slope = return_hyperbolic_function(point_1, point_2, mode, slope)
        y = hyperbolic_f(x)
        y_derivative = hyperbolic_f_derivative(x)
        y_list.append(y)
        y_derivative_list.append(y_derivative)
        slope_list.append(slope)
    hyperbolic_f, hyperbolic_f_derivative, slope = return_hyperbolic_function(point_1, point_2, fixed_point=fixed_point)
    y = hyperbolic_f(x)
    y_derivative = hyperbolic_f_derivative(x)
    y_list.append(y)
    y_derivative_list.append(y_derivative)
    slope_list.append(slope)


    plt.figure()
    plt.title('Hyperbolic functions')
    for i in range(len(y_list)):
        plt.plot(x, y_list[i], label=slope_list[i])
    plt.scatter(fixed_point[0], fixed_point[1], label='Fixed point', s=50)

    plt.ylim((y_min, y_max))
    plt.xlim((x_min, x_max))
    plt.legend()
    plt.show()


    plt.figure()
    plt.title('Hyperbolic functions derivatives')
    for i in range(len(y_derivative_list)):
        plt.plot(x, y_derivative_list[i], label=slope_list[i])

    # plt.xlim((0, x_intercept))
    plt.legend()
    plt.show()

