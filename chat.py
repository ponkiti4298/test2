import streamlit as st
import sqlite3
import sqlite_vss
from json import dumps
from openai import OpenAI

# OpenAI クライアントの初期化
client = OpenAI()

# SQLiteデータベースに接続
@st.cache_resource
def get_database_connection():
    conn = sqlite3.connect("text.db", check_same_thread=False)
    conn.enable_load_extension(True)
    sqlite_vss.load(conn)
    return conn

conn = get_database_connection()

def get_embedding(text, model="text-embedding-3-large", dimensions=1536):
    text = text.replace("\n", " ")
    params = {"model": model, "input": [text], "dimensions": dimensions}
    response = client.embeddings.create(**params)
    return response.data[0].embedding

def find_similar_documents(query, limit=10):
    query_embedding = get_embedding(query)

    search_query = """
    SELECT documents.content, vss_documents.distance
    FROM vss_documents
    JOIN documents ON documents.id = vss_documents.rowid
    WHERE vss_search(vss_documents.embedding, vss_search_params(?, ?))
    ORDER BY vss_documents.distance
    """

    result = conn.execute(search_query, (dumps(query_embedding), limit)).fetchall()
    return result

def generate_response(query, context):
    prompt = f"""以下の情報を参考にして、質問に答えてください。情報:{context}質問: {query}回答:"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=1000,
        stream=True
    )
    return response

st.title("GPT-4o ニュースチャットボット")

# チャット履歴の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []

# チャット履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ユーザー入力の処理
if prompt := st.chat_input("質問を入力してください"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    print(st.session_state.messages)
    
    # 類似文書の検索
    results = find_similar_documents(prompt)
    context = "\n".join([content for content, _ in results])

    # GPT-4を使用した回答の生成
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in generate_response(prompt, context):
            if chunk.choices[0].delta.content is not None:
                full_response += chunk.choices[0].delta.content
                response_placeholder.markdown(full_response + "▌")
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})