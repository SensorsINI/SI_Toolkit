# System Identification Toolkit (SI_Toolkit)

## Overview

The SI_Toolkit is a library designed for easy and effective development of machine learning models for system identification and neural control. It facilitates the training of predictive models and controllers using neural networks (NNs) and provides scripts for data preprocessing, training via supervised learning, testing, and using the models in an autoregressive way. It provides a ready-to-use solution that allows users to focus on application, minimizing the need for custom coding.

## Features

- **Data Analysis Scripts**: Calculation of data statistics summaries and histograms to aid in data inspection before using it for machine learning applications.
- **Model Support**: Basic models like MLP, RNN, GRU, LSTM can be trained using a simple interface with 0 coding using both Tensorflow and Pytorch backends. Custom models – either ML or ODEs with trainable parameters - can be easily added based on provided examples.
- **Brunton Test**: A unique tool that visualizes the predicted evolution of dynamical systems over time, highlighting model accuracy through error visualization. This feature is designed to provide intuitive insights into model performance.
- **Data Processing Utilities**: Comes with features to enrich data, improving model training outcomes through time-shift adjustments, adding derivatives calculated from data, and sensor quantization.
- **Cross-Platform Functionality**: Tested mostly on macOS M1 (Sonoma), it is used successfully on Ubuntu 20.4 and Windows, making it accessible to a wide range of users.
- **FPGA Integration with hls4ml and EdgeDRNN**: Its unique feature is basic integration with hls4ml and the EdgeDRNN framework, which facilitates bringing trained neural networks to FPGA, making it ideal for high-performance, low-latency applications.
- **Framework Agnostic Design**: The SI_Toolkit's architecture is designed to be independent of the underlying machine learning framework. This dual compatibility with TensorFlow and PyTorch not only offers flexibility in model development but also serves as a unique example of creating ML software that allows user to switch between both frameworks while avoiding duplicate code to implement desired functionalities.

## Target Users

The toolkit is crafted for individuals and teams eager to delve into system identification and neural control without the overhead of extensive programming. It was created for use at workshops, but is also a great working example for a pipeline from raw simulated or experimental data to a neural network model deployed on specialized hardware. Together with our other Neural Control Tools, it can substantially speed up the development of ML-based control.

## Extensibility

SI_Toolkit is designed for easy customization and extension. Users can adapt it to fit their specific needs and are invited to contribute to a collaborative and evolving toolkit.

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

    or otherwise, if you rather wanted a separate copy in your repository,
    clone it to your project directory with

    `git clone https://github.com/SensorsINI/SI_Toolkit`

    Notice that as of today,
    Pycharm requires restart to recognize the new submodule,
    and show it within its git tools.

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

## Quick Start - Main Pipeline
1. Collect Training Data
  - To use SI_Toolkit you need a dataset (csv files)
    with columns for input and output of the network you want to train.
  - SI_Toolkit expect to find in experiment folder a data folder (default called `Recordings`)
    and within it three other folders: `Train`, `Validate`, `Test` with data to be used in training and testing.
  - In SI_Toolkit_ASF, we provide an example dataset xxx with data from a cartpole.
    You can use this for the first tests.

For all next steps - The launch scripts are located in `./SI_Toolkit_ASF/Run/`.
They all assume that working directory is the root of the project, and that the SI_Toolkit_ASF folder is in the root of the project.
To do it either run the scripts from terminal from root e.g.
`python SI_Toolkit_ASF/Run/A1_Create_Normalization_File.py`
or `python -m SI_Toolkit_ASF.Run.A1_Create_Normalization_File`

or set the working directory in your IDE to the root of the project.
In Pycharm it can be done in run configurations window.

TODO: Make the (at least starting) working directory more flexible
to be able to run the scripts from where they are located.

2. Calculate data statistics and inspect data
  - This step in necessary for each new dataset, as the data statistics file calculated offline is used while training neural networks to normalize the data.
  - Set paths to experiment folders in `./SI_Toolkit_ASF/config_training.yml`
    - In our example: Set to `./Experiment_Recordings/Dataset-1/`
  - Then run `python -m SI_Toolkit_ASF.Run.Create_normalization_file`
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

## HLS4ML

SI_Toolkit integrates hls4ml and allows for conversion of trained neural networks to FPGA deployable code.
To make the conversion:
1. Install Vivado 2020.1 - at the time of writing, this is the latest version supported by hls4ml.
We recommend you install Vivado together with Vitis - unless you know you need only Vivado,
but our programs need Vitis too.
2. Open config_hls.yml and set the path to Vivado installation.
3. Set the parameters for the network conversion.
4. Optional: Testing the networks quantization effect with Brunton GUI - check the section on Brunton GUI for details.
5. Run the launch script `python -m SI_Toolkit_ASF.Run.Convert_Network_With_hls4ml`




## Gallery
Testing with Brunton GUI predicted vs actual car trajectories
Physical Cartpole swing-up using Model Predictive Control (MPC) with neural network model trained with SI_Toolkit

## More Information
For more detailed information on features and usage, see our [detailed documentation](https://github.com/SensorsINI/SI_Toolkit/wiki/Detailed-Documentation).