import sqlite3

# Conectar ao banco de dados (ou criar se não existir)
conn = sqlite3.connect('face_recognition.db')
cursor = conn.cursor()

# Criar a tabela para armazenar as informações das pessoas
cursor.execute('''
CREATE TABLE IF NOT EXISTS people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    sus_card TEXT NOT NULL,
    face_encoding BLOB NOT NULL
)
''')

conn.commit()
conn.close()
