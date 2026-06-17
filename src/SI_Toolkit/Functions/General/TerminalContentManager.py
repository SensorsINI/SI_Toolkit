import sys
import os
import re
import shutil


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
            if self.buffer:
                self.clear_temporary_messages()
                printed_buffer = self.print_buffered_messages()
                if printed_buffer:
                    self.terminal.write('\n')
                self.print_temporary_messages()
            else:
                self.overwrite_temporary_messages()
            self.terminal.flush()

    def clear_temporary_messages(self):
        ESC = '\033['
        CLEAR_LINE = ESC + '2K'  # Clear the entire line

        for _ in range(self.counter_temporary_messages):
            self.terminal.write('\r' + ESC + '1A' + CLEAR_LINE)

        self.counter_temporary_messages = 0

    def print_buffered_messages(self):
        buffered_text = ''.join(self.buffer)
        self.buffer.clear()

        if not buffered_text:
            return False

        self.terminal.write(buffered_text)
        if not buffered_text.endswith('\n'):
            self.terminal.write('\n')

        return True

    def print_temporary_messages(self):
        self.counter_temporary_messages = 0
        for message in self.buffer_temporary:
            self.counter_temporary_messages += count_terminal_rows(message)
            self.terminal.write(message)
        self.buffer_temporary.clear()

    def overwrite_temporary_messages(self):
        old_rows = self.counter_temporary_messages
        new_rows = sum(count_terminal_rows(message) for message in self.buffer_temporary)

        if old_rows:
            self.terminal.write(f'\033[{old_rows}A')

        for message in self.buffer_temporary:
            self.terminal.write(message)
        self.buffer_temporary.clear()

        extra_rows = old_rows - new_rows
        if extra_rows > 0:
            for _ in range(extra_rows):
                self.terminal.write('\r\033[2K\n')
            self.terminal.write(f'\033[{extra_rows}A')

        self.counter_temporary_messages = new_rows


def count_terminal_rows(text):
    terminal_width = shutil.get_terminal_size(fallback=(120, 24)).columns
    visible_text = strip_escape_sequences(text).replace('\r', '')
    rows = 0

    for line in visible_text.splitlines():
        rows += max(1, (len(line) + terminal_width - 1) // terminal_width)

    return rows


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


