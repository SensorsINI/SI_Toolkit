# SI_Toolkit
System identification with neural networks - training and testing scripts

Use this repository to train neural networks on dynamical systems.

1. Collect Training Data
  - Run a data generation script on your simulation environment.
  - Example: run_data_generator in CartPoleSimulator
2. Load Training Data and Normalize it
  - Set `folder_with_data_to_calculate_norm_info` in `config.yml`
  - Then run:
  ```bash
  python3 -m SI_Toolkit.load_and_normalize
  ```
  - This creates a new file in `SI_Toolkit/NormalizationInfo/`
3. Train Network
  - Type `python3 -m SI_Toolkit.TF.Train -h` to view all arguments that can be passed to training script
4. View model performance
  - Run script `run_brunton_test.py` to see how your model performs on test data
  - All parameters for this script are in the 'testing' section of the config
5. Run simulator with trained network model
  - Specify `NET_NAME` in config -> this is the network used in `predictor_autoregressive.py`
  - When using the CartPoleSimulator repo as system: In the CartPoleSimulator config, specify "NeuralNetwork" as predictor type
