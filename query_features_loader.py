#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:28:13 2024

@author: sohail
"""

import sqlite3

class QueryFeaturesLoader:
    def __init__(self, db_path):
        # Connect to the SQLite database
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()

        # Query all rows from the database (adjust query if needed)
        self.cursor.execute("SELECT * FROM TestQueryData")
        self.rows = self.cursor.fetchall()
        self.current_index = 0  # Keeps track of the current row being accessed
        

    def get_query_features(self):
        # If we've exhausted all rows, return None
        if self.current_index >= len(self.rows):
            self.close()
            return None
    
        # Extract features for the current query from the database row
        row = self.rows[self.current_index]
        query_features = {
            'QueryString': row[0],           # QueryString TEXT
            'row_id': self.current_index + 1,  # Index-based row ID
            'QueryVars': row[1],             # QueryVars TEXT
            'ProjVars': row[2],              # ProjVars TEXT
            'JoinVars': row[3],              # JoinVars TEXT
            'NonJoinVars': row[4],           # NonJoinVars TEXT
            'MinCardinalityInTP': row[5],    # MinCardinalityInTP TEXT
            'SelectModifier': row[6],        # SelectModifier INT
            'NoTPs': row[7],                 # NoTPs INT
            'TPSizes': row[8],               # TPSizes TEXT
            'VartoLabelMap': row[9],         # VartoLabelMap TEXT (new column)
            'QueryPlan': row[10],            # QueryPlan TEXT
            'TentrisQueryRuntime': row[11],  # TentrisQueryRuntime REAL
            'DRLQueryRuntime': row[12]       # DRLQueryRuntime REAL
        }
    
        # Increment the index for the next query
        self.current_index += 1
    
        return query_features

    def close(self):
        # Close the SQLite connection when done
        self.conn.close()
        
    def __del__(self):
        self.close()  # Close the connection when the object is deleted