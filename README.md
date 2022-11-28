# SI_Toolkit
System identification with neural networks - training and testing scripts

Use this repository to train neural networks on dynamical systems.

1. Collect Training Data
  - Run a data generation script on your simulation environment.
  - Example: run_data_generator in CartPoleSimulator
  - Or create a folder `./Experiment_Recordings/Dataset-1/` and put data in there
2. Load Training Data and Normalize it
  - Set paths to experiment folders in `./SI_Toolkit_ASF/config_training.yml`
    - In our example: Set to `./Experiment_Recordings/Dataset-1/`
  - Then run `python -m SI_Toolkit_ASF.run.Create_normalization_file`
  - This creates a new normalization file in `[path_to_experiment_folder]/NormalizationInfo/`
3. Train Network
  - Type `python -m SI_Toolkit_ASF.run.Train_Network -h` to view all arguments that can be passed to training script
  - You will want to specify a `--net_name` (see help message to see how to do that)
  - Specify the `--path_to_normalization_info` to be the CSV file created in step 2.
  - Training creates a new folder with the network name in `[path_to_experiment_folder]/Models/` which contains
    - Save checkpoints
    - A `.txt` file to preserve all the params used for training
    - A `.csv` file and plot to document the training progress / loss curves
4. View model performance
  - Adjust the parameters in `./SI_Toolkit_ASF/config_testing.yml` to select the correct test file and model
  - Run script `python -m SI_Toolkit_ASF.run.Run_Brunton_Test` to see how your model performs on test data
  - All parameters for this script are in the 'testing' section of the config:
    - `PATH_TO_NORMALIZATION_INFO`: Path to the normalization info file created in 2.
    - `tests`: A list of model names, typically just the one model trained in 3.
    - `TEST_FILE`: An experiment run to compare observations and predictions on. Specify any list of CSVs generated in 1.
5. Run simulator with trained network model
  - Specify `NET_NAME` in config -> this is the network used in `predictor_autoregressive_neural.py`
  - When using the CartPoleSimulator repo as system: In the CartPoleSimulator config, specify `"predictor_autoregressive_neural"` as predictor type
