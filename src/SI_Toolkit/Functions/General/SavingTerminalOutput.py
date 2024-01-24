import sys


class DualOutput:
    def __init__(self, filename, terminal):
        self.terminal = terminal
        self.log = open(filename, "a")
        self.progress_bar_state = ""

    def write(self, message):
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
        self.terminal.flush()
        self.log.flush()




class TerminalSaver:
    def __init__(self, filename):
        self.filename = filename
        self.terminal = sys.stdout

    def __enter__(self):
        sys.stdout = DualOutput(self.filename, self.terminal)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.flush()
        sys.stdout = self.terminal
