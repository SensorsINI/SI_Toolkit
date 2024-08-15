from SI_Toolkit.LivePlotter.live_plotter_sender import LivePlotter_Sender
import time
import pandas as pd
from tqdm import tqdm


USE_REMOTE = False

REMOTE_USERNAME = 'marcinpaluch'
REMOTE_IP = '192.168.194.233'
DEFAULT_ADDRESS = ('localhost', 6000)

def main():

    sender = LivePlotter_Sender(DEFAULT_ADDRESS, USE_REMOTE, REMOTE_USERNAME, REMOTE_IP)

    sender.connect()

    while not sender.connection_ready:
        time.sleep(0.1)

    path = 'Experiment.csv'
    # path = '../../../ExperimentRecordings/CPP_mpc__2024-07-01_00-52-52.csv'
    df = pd.read_csv(path, comment='#')
    df = df[['time', 'angle', 'angleD', 'Q', 'Q_applied']]

    try:
        # Send the header
        header = df.columns.tolist()
        sender.send_headers(header)
        time.sleep(0.1)

        # Send data line by line
        for _, row in tqdm(df.iterrows()):
            sender.send_data(row.values)
            time.sleep(0.02)  # Wait for a short time before sending the next row

        # Optionally, send a 'complete' message to indicate end of data
        sender.send_complete()

    finally:
        sender.close()


if __name__ == '__main__':
    main()
