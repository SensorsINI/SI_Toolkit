# SI_Toolkit
System identification with neural networks - training and testing scripts

Use this repository to train neural networks on dynamical systems.

1. Collect Training Data
  - Run a data generation script on your simulation environment.
  - Example: run_data_generator in CartPoleSimulator
  - For example, create a folder `./Experiment_Recordings/Dataset-1/`
2. Load Training Data and Normalize it
  - Set `folder_with_data_to_calculate_norm_info` in `config.yml`
    - In our example: Set to `./Experiment_Recordings/Dataset-1/`
  - Then run:
  ```bash
  python3 -m SI_Toolkit.load_and_normalize
  ```
  - This creates a new normalization file in `SI_Toolkit/NormalizationInfo/`
3. Train Network
  - Split experiment recording files into subfolders `Train`, `Validate`, `Test` within `./Experiment_Recordings/Dataset-1/` (currently done manually).
  - Type `python3 -m SI_Toolkit.TF.Train -h` to view all arguments that can be passed to training script
  - You will want to specify a `--net_name` (see help message to see how to do that)
  - Specify the `--path_to_normalization_info` to be the CSV file created in step 2.
  - Training creates a new folder with the network name in `SI_Toolkit/TF/Model/` which contains
    - Save checkpoints
    - A `.txt` file to preserve all the params used for training
    - A `.csv` file and plot to document the training progress / loss curves
4. View model performance
  - Run script `python3 -m SI_Toolkit.Testing.run_brunton_test` to see how your model performs on test data
  - All parameters for this script are in the 'testing' section of the config:
    - `PATH_TO_NORMALIZATION_INFO`: Path to the normalization info file created in 2.
    - `tests`: A list of model names, typically just the one model trained in 3.
    - `TEST_FILE`: An experiment run to compare observations and predictions on. Specify any list of CSVs generated in 1.
5. Run simulator with trained network model
  - Specify `NET_NAME` in config -> this is the network used in `predictor_autoregressive.py`
  - When using the CartPoleSimulator repo as system: In the CartPoleSimulator config, specify `"NeuralNet"` as predictor type
