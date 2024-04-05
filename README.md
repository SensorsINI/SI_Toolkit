# SI_Toolkit
System Identification with neural networks and GPs with scripts for training, testing and related data processing.

The Toolkit was tested primarily with conda python 3.11 on macOS M1 (Sonoma),
but is used without problems on Ubuntu 20.4 and to lesser extent on Windows.

## Installation
These steps are needed only if you create a new project.

Our projects which use the Toolkit have it as a submodule and it is cloned and  installed automatically.
If you create a new project:

0. Create python 3.11 environment, if you haven't done it yet. e.g.

    `conda create -n myenv python=3.11`

    `conda activate myenv`

    `conda install pip`

1. Clone the repository

    SI_Toolkit is an installable library to be used as part of bigger project.
    we recommend that you add it as a submodule at the root of your project hierarchy with

    `git submodule add https://github.com/SensorsINI/SI_Toolkit`

    or otherwise clone it to your project directory with

    `git clone https://github.com/SensorsINI/SI_Toolkit`

2. Install the Toolkit

    `pip install -e ./SI_Toolkit`

    This line assumes that you cloned the Toolkit to the root of your project hierarchy, otherwise adjust the path.
    -e flag installs the Toolkit in editable mode, so you can modify the Toolkit and see the changes in your project without reinstalling it.
    This installation also installs all the dependencies - it is quite a lot of packages, so it may take a while.
3. Copy SI_Toolkit_ASF_Template to the root of your project hierarchy and rename it to SI_Toolkit_ASF, e.g.

    `cp -r ./SI_Toolkit/SI_Toolkit_ASF_Template ./SI_Toolkit_ASF`

   The ASF stand for Application Specific files.
   This folders contains the files that need to be customized for your project- mostly configuration files, run scripts and default output locations.
   We discuss the customization of these files in the sections related to functionalities of interest.

## Main Pipeline
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

## Using a custom model
To train more complex models, you can write your own:
1. Write your module `SI_Toolkit_ASF/Modules/[module_name].py`
  - Write a class that inherits from `tf.keras.Model`
  - The class must have a `__init__(self)` function and a `call(self, x)` function. `x` is a tensor containing the input to the network. Refer to https://keras.io/api/models/model/ for more information on how to write your own model.
2. For training, follow the steps outlined above. 
  - Use your custom model by specifying `--net-name Custom-[module_name]-[class_name]`