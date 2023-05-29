import numpy as np


def return_hyperbolic_function(x_intercept, y_intercept, mode=1, slope=None, fixed_point=None):
    if slope is None and fixed_point is None:
        raise ValueError('slope and fixed_point cannot be both None!')

    if fixed_point is None:
        a = slope
        delta_b = x_intercept ** 2 + 4 * (x_intercept / y_intercept) * a
        b = (-x_intercept + mode * np.sqrt(delta_b)) / 2
        c = y_intercept - (a / b)
    else:
        x_target, y_target = fixed_point[0], fixed_point[1]
        A = x_target*y_target/(1-(x_target/x_intercept)-(y_target/y_intercept))
        b = A/y_intercept
        c = -A/x_intercept
        a = A-b*c

    def hyperbolic_function(x):
        return a/(x+b) + c

    def hyperbolic_function_derivative(x):
        return a/((x+b)**2)

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

    a = [0.1, 1.0, 100.0]
    x_intercept = 1.0
    y_intercept = 1.0
    mode = 1
    fixed_point = (0.4, 0.8)

    x = np.linspace(0.0, x_intercept, 1000)

    y_list = []
    y_derivative_list = []
    slope_list = []
    for slope in a:
        hyperbolic_f, hyperbolic_f_derivative, slope = return_hyperbolic_function(x_intercept, y_intercept, mode, slope)
        y = hyperbolic_f(x)
        y_derivative = hyperbolic_f_derivative(x)
        y_list.append(y)
        y_derivative_list.append(y_derivative)
        slope_list.append(slope)
    hyperbolic_f, hyperbolic_f_derivative, slope = return_hyperbolic_function(x_intercept, y_intercept, fixed_point=fixed_point)
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

    plt.ylim((0, y_intercept))
    plt.xlim((0, x_intercept))
    plt.legend()
    plt.show()


    plt.figure()
    plt.title('Hyperbolic functions derivatives')
    for i in range(len(y_derivative_list)):
        plt.plot(x, y_derivative_list[i], label=slope_list[i])

    plt.xlim((0, x_intercept))
    plt.legend()
    plt.show()

