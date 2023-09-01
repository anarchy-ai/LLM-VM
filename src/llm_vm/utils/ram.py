import psutil
import sys

class RAMLogger:
    AMBER = '\033[93m'
    END_COLOR = '\033[0m'

    def __init__(self):
        self.start_ram = None
        self.end_ram = None

    def start(self):
        self.start_ram = self.get_ram_usage()
        print(f"{self.AMBER} RAM Usage:{self.END_COLOR}", file=sys.stderr)
        self.print_progress_bar(self.start_ram)

    def end(self):
        self.end_ram = self.get_ram_usage()
        print(f"{self.AMBER}Ending RAM Usage:{self.END_COLOR}", file=sys.stderr)
        self.print_progress_bar(self.end_ram)

    @staticmethod
    def get_ram_usage():
        return psutil.virtual_memory().percent

    def print_progress_bar(self, percentage):
        bar_length = 50
        block = int(round(bar_length * percentage / 100))
        progress = "|" + "█" * block + "-" * (bar_length - block) + "|"
        print(f"{self.AMBER}{progress} {percentage}%{self.END_COLOR}", file=sys.stderr)
