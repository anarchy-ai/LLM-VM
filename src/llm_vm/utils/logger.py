import sqlite3
import time

def log_telemetry(logSource, logLevel, logMessage, logData, logType):
    conn = sqlite3.connect('anarchy.db')
    cursor = conn.cursor()
    
    timestamp = int(time.time())
    
    cursor.execute('INSERT INTO telemetrics (timestamp, logSource, logLevel, logMessage, logData, logType) VALUES (?, ?, ?, ?, ?, ?)', (timestamp, logSource, logLevel, logMessage, logData, logType))
    
    conn.commit()
    conn.close()