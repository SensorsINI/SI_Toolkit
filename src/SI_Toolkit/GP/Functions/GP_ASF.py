import gpflow as gpf


def get_kernels(inputs):

    indices = {key: inputs.index(key) for key in inputs}

    kernels = {"position": gpf.kernels.RBF(lengthscales=[1, 1, 1],
                                           active_dims=[indices["position"],
                                                        # indices["angleD"],
                                                        indices["positionD"],
                                                        indices["Q"]
                                                        ]),

               "positionD": gpf.kernels.RBF(lengthscales=[1, 1],
                                            active_dims=[# indices["angle_sin"],
                                                         # indices["angle_cos"],
                                                         # indices["angleD"],
                                                         indices["positionD"],
                                                         indices["Q"]
                                                         ]),

               "angle_sin": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1, 1],
                                            active_dims=[indices["angle_sin"],
                                                         indices["angle_cos"],
                                                         indices["angleD"],
                                                         indices["positionD"],
                                                         indices["Q"]
                                                         ]),

               "angle_cos": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1, 1],
                                            active_dims=[indices["angle_sin"],
                                                         indices["angle_cos"],
                                                         indices["angleD"],
                                                         indices["positionD"],
                                                         indices["Q"]
                                                         ]),

               "angleD": gpf.kernels.RBF(lengthscales=[1, 1, 1, 1, 1],
                                                 active_dims=[indices["angle_sin"],
                                                              indices["angle_cos"],
                                                              indices["angleD"],
                                                              indices["positionD"],
                                                              indices["Q"]
                                                              ])

    }

    return kernels