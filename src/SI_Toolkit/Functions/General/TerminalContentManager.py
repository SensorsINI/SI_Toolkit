import sys
import os
import re


class DualOutput:
    def __init__(self, filename=None, terminal=None, special_print_function=False):
        self.terminal = terminal if terminal else sys.stdout
        self.special_print_function = special_print_function
        self.progress_bar_state = ""
        self.buffer = []  # Buffer to accumulate messages
        self.buffer_temporary = []  # Buffer to accumulate temporary messages
        self.counter_temporary_messages = 0

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
                # Write regular messages to both file and buffer
                self.log.write(strip_escape_sequences(message))

        # Always write to the terminal (both progress bar updates and regular messages)
        if not self.special_print_function:
            self.terminal.write(message)
        else:
            self.buffer.append(message)  # Accumulate messages in the buffer

    def flush(self):
        pass

    def print_to_terminal(self):
        """
        This is used if special_print_function is set to True.
        """
        if self.special_print_function:
            ESC = '\033['
            CLEAR_LINE = ESC + 'K'  # Clear the entire line after the cursor

            # Clear the lines with temporary messages
            for _ in range(self.counter_temporary_messages):
                self.terminal.write(ESC + '1A' + CLEAR_LINE)  # Move cursor up and clear the line
            self.terminal.flush()

            # Print accumulated messages to the terminal
            for message in self.buffer:
                self.terminal.write(message)
            self.buffer.clear()  # Clear the buffer after printing
            self.terminal.flush()

            # Get new counter value
            # Print temporary messages to the terminal
            self.counter_temporary_messages = 0
            for message in self.buffer_temporary:
                newline_count = message.count('\n')
                self.counter_temporary_messages += newline_count
                self.terminal.write(message)
            self.buffer_temporary.clear()

            self.terminal.flush()

def strip_escape_sequences(text):
    # Regular expression to match ANSI escape sequences
    ansi_escape = re.compile(r'(?:\x1B[@-_][0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)


class TerminalContentManager:
    def __init__(self, filename=None, special_print_function=False):
        self.filename = filename
        self.terminal = sys.stdout
        self.my_stdout = DualOutput(self.filename, self.terminal, special_print_function)

    def __enter__(self):
        sys.stdout = self.my_stdout
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.my_stdout.terminal.flush()
        if self.my_stdout.log is not None:
            self.my_stdout.log.flush()
            self.my_stdout.log.close()
        sys.stdout = self.terminal

    def print_to_terminal(self):
        self.my_stdout.print_to_terminal()

    def print_temporary(self, message, end='\n'):
        self.my_stdout.buffer_temporary.append(message + end)


