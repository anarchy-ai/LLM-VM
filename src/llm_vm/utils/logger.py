import sqlite3
import time

def log_telemetry(logSource, logLevel, logMessage, logData, logType):
    """
    Inserts a new log entry into the 'telemetrics' table in the 'anarchy.db' SQLite database.
    
    Parameters:
        logSource (str): Source of the log
        logLevel (str): Level of the log (e.g., 'INFO', 'ERROR', 'WARNING')
        logMessage (str): Log message text
        logData (str): Additional log data
        logType (str): Type of the log (e.g., 'SystemLog', 'UserLog')

    Returns:
        None
    """
    conn = sqlite3.connect('anarchy.db')
    cursor = conn.cursor()
    
    timestamp = int(time.time())
    
    cursor.execute('INSERT INTO telemetrics (timestamp, logSource, logLevel, logMessage, logData, logType) VALUES (?, ?, ?, ?, ?, ?)', (timestamp, logSource, logLevel, logMessage, logData, logType))
    
    conn.commit()
    conn.close()