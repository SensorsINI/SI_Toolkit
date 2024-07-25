# Live Plotter

## Overview

Live Plotter is a tool for real-time data plotting and visualization. It consists of a server-side component that handles data reception and plotting, and a client-side component that sends data to the server for visualization.

## Structure

### Receiver (Server) Side: Plotting the Data

1. **LivePlotter_ConnectionHandlerReceiver** (`live_plotter_x_connection_handler_receiver`):
   - Manages communication with the client, which sends data and possibly user control commands.

2. **LivePlotter** (`live_plotter.py`):
   - Receives data and user control commands from `LivePlotter_ConnectionHandlerReceiver` and user control commands from the GUI (`live_plotter_gui.py`).
   - Passes data and commands to `LivePlotter_Plotter`.
   - Handles visualization update rate using Matplotlib animation.

3. **LivePlotter_Plotter** (`live_plotter_plotter.py`):
   - Manages the actual plotting tasks using Matplotlib.

#### How to Start the Server

- **Option 1**: Run `live_plotter_gui.py` to start the server with a GUI for user control.
- **Option 2**: Run `live_plotter.py` to display the plot only, without GUI, with limited options for user control.

> **Recommendation**: Use the launching script `SI_Toolkit_ASF/Run/Run_LivePlotter.py` for easier startup.

### Sender (Client) Side: Sending the Data

1. **LivePlotter_ConnectionHandlerSender** (`live_plotter_x_connection_handler_sender`):
   - Manages communication with the server, which handles the plotting of the data.

2. **LivePlotter_Sender** (`live_plotter_sender.py`):
   - Provides a user-friendly interface for sending data and possibly user control commands through convenient methods.

#### Example Usage

- **Example Script**: `xxx_live_plotter_sending_testdata.py`
  - Demonstrates how to use `LivePlotter_Sender` in your environment.
  - Can be used for testing with the provided file `Experiment.csv`.
  - To run the example, set the working directory to the directory containing `Experiment.csv` and run the script.

> **Note**: Ensure the server side is started before running the client script. The example script includes a built-in delay to facilitate simultaneous startup, such as using PyCharm's "Compound run configurations".

