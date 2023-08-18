import sqlite3
import time

def initialize_db():
    print('Initialising database...')
    conn = sqlite3.connect('anarchy.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS telemetrics (
            id INTEGER PRIMARY KEY,
            timestamp INTEGER,
            logSource TEXT,
            logLevel TEXT,
            logMessage TEXT,
            logData TEXT,
            logType TEXT
        )
    ''')

    cursor.execute('SELECT * FROM telemetrics WHERE logMessage = "db initialized"')
    if cursor.fetchone() is None:
        timestamp = int(time.time())
        cursor.execute('INSERT INTO telemetrics (timestamp, logSource, logLevel, logMessage, logData, logType) VALUES (?, ?, ?, ?, ?, ?)',
                       (timestamp, 'db_initializer', 'INFO', 'db initialized', 'db initialized successfully', 'SystemLog'))
        
    conn.commit()
    conn.close()
