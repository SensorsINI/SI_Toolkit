import subprocess
import threading
import time
from multiprocessing.connection import Client


class LivePlotter_ConnectionHandlerSender:
    def __init__(self, address, remote_username=None, remote_ip=None, callback=None):
        self.address = address
        self.remote_username = remote_username
        self.remote_ip = remote_ip

        self.ssh_process = None
        if remote_username and remote_ip:
            self.use_remote = True
        else:
            self.use_remote = False

        self._connection_ready = False

        self.connection = None
        self.callback = callback  # Callback to notify when connection is ready

    def establish_ssh_tunnel(self):
        # Define the SSH command to establish the tunnel
        ssh_command = [
            'ssh',
            '-L', f"{self.address[1]}:{self.address[0]}:{self.address[1]}",  # Local port to remote port
            f"{self.remote_username}@{self.remote_ip}"  # Username and remote host
        ]

        # Start the SSH tunnel as a subprocess
        process = subprocess.Popen(ssh_command)
        return process

    def connect(self):
        thread = threading.Thread(target=self._connect)
        thread.start()

    def _connect(self):
        try:
            if self.use_remote:
                self.ssh_process = self.establish_ssh_tunnel()
                time.sleep(5)  # Simulate waiting for SSH tunnel to establish
            else:
                time.sleep(1)  # Simulate waiting for local setup

            self.connection = Client(self.address)
            # Check connection by sending a ping
            if self._check_connection():
                self.connection_ready = True
            else:
                self.connection_ready = False
                self.connection.close()
        except Exception as e:
            print(f"Failed to connect: {e}")
            self.connection_ready = False

    def _check_connection(self):
        try:
            self.connection.send("ping")
            response = self.connection.recv()
            if response == "pong":
                return True
            print('Could not establish connection to Live Plotter Server.')
            return False
        except Exception as e:
            print(f"Connection check failed: {e}")
            print(f"Is your Live Plotter Server running?")
            print(f"Is your address correct and SSH tunnel established (if needed)?")
            return False

    @property
    def connection_ready(self):
        return self._connection_ready

    @connection_ready.setter
    def connection_ready(self, value):
        self._connection_ready = value
        if value and self.callback:
            self.callback(value)  # Call the callback when connection is ready

    def send(self, data):
        if not self.connection_ready:
            raise Exception("Connection not established yet.")
        self.connection.send(data)

    def close(self):
        if self.connection:
            self.connection.close()  # Close the connection
            self.connection_ready = False

        if self.use_remote and self.ssh_process:
            # Terminate the SSH tunnel
            self.ssh_process.terminate()
            self.ssh_process.wait()  # Wait for the SSH process to terminate
