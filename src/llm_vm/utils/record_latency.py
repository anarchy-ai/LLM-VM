import time
from llm_vm.utils.logger import log_telemetry
from llm_vm.utils.stdlog import setup_logger

class RecordLatency:
    def __init__(self, logSource="llm_vm"):
        self.start_time = None
        self.end_time = None
        self.latency = None
        self.logSource = logSource
        self.logLevel = "info"
        self.logType = "latency"
    
    def start(self):
        self.start_time = time.time()
    
    def end(self):
        logger = setup_logger(__name__)
        self.end_time = time.time()
        if self.start_time is None:
            logger.error("Start time was not set. Call the 'start' method first.")
            return
        self.latency = self.end_time - self.start_time
        # Log the latency using the log_telemetry function
        logMessage = f'Recorded Latency: {self.latency:.6f} seconds'
        logData = None
        log_telemetry(self.logSource, self.logLevel, logMessage, logData, self.logType)

    def get_latency(self):
        return self.latency