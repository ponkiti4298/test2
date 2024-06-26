import sqlite3
import sqlite_vss
from json import dumps
from openai import OpenAI
import pandas as pd

# OpenAI クライアントの初期化
client = OpenAI()

# SQLiteデータベースに接続
conn = sqlite3.connect("text.db")
cursor = conn.cursor()

# vector0とvss0拡張をロード
conn.enable_load_extension(True)
sqlite_vss.load(conn)

# テーブルを作成（存在しない場合）
cursor.execute(
"""
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY,
    content TEXT,
    embedding BLOB
)
"""
)

cursor.execute(
    "CREATE VIRTUAL TABLE IF NOT EXISTS vss_documents USING vss0(embedding(1536))"
)


def get_embedding(text, model="text-embedding-3-large", dimensions=1536):
    text = text.replace("\n", " ")
    params = {"model": model, "input": [text], "dimensions": dimensions}
    response = client.embeddings.create(**params)
    return response.data[0].embedding


def insert_document(content):
    embedding = get_embedding(content)

    # documentsテーブルに挿入
    cursor.execute(
        "INSERT INTO documents (content, embedding) VALUES (?, ?)",
        (content, dumps(embedding)),
    )
    doc_id = cursor.lastrowid

    # vss_documentsテーブルに挿入
    cursor.execute(
        "INSERT INTO vss_documents (rowid, embedding) VALUES (?, ?)",
        (doc_id, dumps(embedding)),
    )

    conn.commit()


# Excelファイルを読み込む
excel_file = "text.xlsx"  # Excelファイルのパスを指定
df = pd.read_excel(excel_file)

# text列の各行をベクトル化してSQLiteに保存
for text in df['text']:
    insert_document(text)

print("すべてのドキュメントが正常に挿入されました。")

# データベース接続を閉じる
conn.close()