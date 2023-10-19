import psutil
import sys

class DiskUsage:
    AMBER = '\033[93m'
    END_COLOR = '\033[0m'

    def __init__(self):
        self.start_disk = None
        self.end_disk = None

    def start(self):
        self.start_disk = self.get_disk_usage()
        print(f"{self.AMBER} Disk Usage:{self.END_COLOR}", file=sys.stderr)
        self.print_progress_bar(self.start_disk)

    def end(self):
        self.start_disk = self.get_disk_usage()
        print(f"{self.AMBER}Ending Disk Usage:{self.END_COLOR}", file=sys.stderr)
        self.print_progress_bar(self.end_disk)

    @staticmethod
    def get_disk_usage():
        percent = 0
        print(psutil.disk_partitions())
        for partition in psutil.disk_partitions():
            dis_usage = psutil.disk_usage(partition.mountpoint)
            percent = percent + dis_usage.percent
        return percent

    def print_progress_bar(self, percentage):
        bar_length = 50
        block = int(round(bar_length * percentage / 100))
        progress = "|" + "█" * block + "-" * (bar_length - block) + "|"
        print(f"{self.AMBER}{progress} {percentage}%{self.END_COLOR}", file=sys.stderr)

