class PredictorWrapper:
    def __init__(self, predictors_config):

        self.horizon = None
        self.batch_size = None

        self.predictor = None

        self.predictor_name = predictors_config['predictor_name_main']
        self.predictor_config = predictors_config['predictors'][self.predictor_name]
        self.predictor_type = predictors_config['predictors'][self.predictor_name]['predictor_type']
        self.model_name = predictors_config['predictors'][self.predictor_name]['model_name']

    def initialize(self, batch_size, horizon, compile_standalone=False):

        compile_standalone = {'disable_individual_compilation': not compile_standalone}

        self.batch_size = batch_size
        self.horizon = horizon

        if self.predictor_type == 'neural':
            from SI_Toolkit.Predictors.predictor_autoregressive_tf import predictor_autoregressive_tf
            self.predictor = predictor_autoregressive_tf(horizon=self.horizon, batch_size=self.batch_size, **self.predictor_config, **compile_standalone)

        elif self.predictor_type == 'GP':
            from SI_Toolkit.Predictors.predictor_autoregressive_GP import predictor_autoregressive_GP
            self.predictor = predictor_autoregressive_GP(horizon=self.horizon, batch_size=self.batch_size, **self.predictor_config, **compile_standalone)

        elif self.predictor_type == 'ODE':
            from SI_Toolkit.Predictors.predictor_ODE import predictor_ODE
            self.predictor = predictor_ODE(horizon=self.horizon, batch_size=self.batch_size, **self.predictor_config)

        elif self.predictor_type == 'ODE_TF':
            from SI_Toolkit.Predictors.predictor_ODE_tf import predictor_ODE_tf
            self.predictor = predictor_ODE_tf(horizon=self.horizon, batch_size=self.batch_size, **self.predictor_config, **compile_standalone)

        else:
            raise NotImplementedError('Type of the predictor not recognised.')


    def initialize_with_compilation(self, batch_size, horizon):
        """
        To get max performance
        use this for standalone predictors (like in Brunton test)
        but not predictors within controllers
        """
        self.initialize(batch_size, horizon, compile_standalone=True)

    def predict(self, s, Q):
        return self.predictor.predict(s, Q)

    def update(self, Q0, s):
        if self.predictor_type == 'neural':
            self.predictor.update_internal_state_tf(s=s, Q0=Q0)


