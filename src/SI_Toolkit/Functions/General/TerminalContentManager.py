import sys
import os
import io

class DualOutput:
    def __init__(self, filename=None, terminal=None, save_at_exit=False,):
        self.terminal = terminal if terminal else sys.stdout
        self.buffer = io.StringIO()
        self.progress_bar_state = ""

        if filename:
            try:
                os.makedirs(os.path.dirname(filename), exist_ok=True)
            except FileExistsError:
                pass
            self.log = open(filename, "a")
        else:
            self.log = None

    def write(self, message):
        if self.log is not None:
            if "\r" in message:
                # Update progress bar state, but don't write to file yet
                self.progress_bar_state = message
            else:
                # Write the final progress bar state to the file
                if self.progress_bar_state:
                    self.log.write(self.progress_bar_state + '\n')  # Ensure newline
                    self.progress_bar_state = ""
                # Write regular messages to both file and terminal
                self.log.write(message)

        # Always write to the terminal (both progress bar updates and regular messages)
        self.terminal.write(message)

    def flush(self):
        pass


class TerminalContentManager:
    def __init__(self, filename):
        self.filename = filename
        self.terminal = sys.stdout
        self.my_stdout = DualOutput(self.filename, self.terminal)

    def __enter__(self):
        sys.stdout = self.my_stdout
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.my_stdout.terminal.flush()
        if self.my_stdout.log is not None:
            self.my_stdout.log.flush()
            self.my_stdout.log.close()
        sys.stdout = self.terminal
