import sqlite3

DATABASE = '../database/database.db'

def fetch(query):

    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    cursor.execute(query)
    
    return cursor.fetchall() 
