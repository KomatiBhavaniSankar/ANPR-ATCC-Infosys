import sqlite3
import pandas as pd
import threading
import os

class TrafficDB:
    """
    A simple, thread-safe SQLite database class for logging traffic analysis results.
    """
    def __init__(self, db_name="traffic_analysis.db"):
        self.db_name = db_name
        self.lock = threading.Lock()
        self._initialize_db()

    def _get_connection(self):
        """Returns a new database connection for the current thread."""
        # Check thread identity to manage connections correctly if necessary, 
        # but for simplicity in Streamlit, connecting on call is usually safer.
        return sqlite3.connect(self.db_name)

    def _initialize_db(self):
        """Creates the necessary table if it doesn't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    source_type TEXT,
                    vehicle_class TEXT,
                    count INTEGER,
                    traffic_level TEXT
                )
            """)
            conn.commit()

    def save_result(self, timestamp, source_type, vehicle_class, count, traffic_level):
        """Inserts a single analysis result record."""
        # The Python call site provides 5 values: timestamp, source_type, vehicle_class, count, traffic_level
        with self.lock:
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    # FIX: Reduced the number of placeholders to 5 to match the 5 values passed.
                    cursor.execute("""
                        INSERT INTO analysis_results (timestamp, source_type, vehicle_class, count, traffic_level)
                        VALUES (?, ?, ?, ?, ?) 
                    """, (timestamp, source_type, vehicle_class, count, traffic_level))
                    conn.commit()
            except Exception as e:
                # IMPORTANT: In a debugging context, log the error so you can see it in the console!
                print(f"DB Error saving result: {e}")
                # Pass silently only if absolutely necessary in production
                pass 

    def fetch_all_data(self):
        """Fetches all data from the analysis_results table."""
        with self._get_connection() as conn:
            return pd.read_sql_query("SELECT * FROM analysis_results", conn)

    def clear_db(self):
        """Clears all data from the analysis_results table (for testing/reset)."""
        with self.lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM analysis_results")
                conn.commit()
