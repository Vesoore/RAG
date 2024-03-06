import streamlit as st
from langchain_community.document_loaders.pdf import PyPDFLoader
from Rag import Rag


st.header("Responses by pdf file")
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
if uploaded_file is not None:
    file_contents = uploaded_file.read()
    with open("temp_file.pdf", "wb") as f:
        f.write(file_contents)
rag = Rag()
with st.sidebar:
    rad_llm = st.radio(
        "Choose a llm",
        ("YandexGpt", "Gigachat lite")
    )

    if rad_llm == "YandexGpt":
        api_key = st.text_input("Add your yandex_api_key", type="password")
        folder_id = st.text_input("Add your yandex_folder_id", type="password")
        rag.model(yc_api_key=api_key, yc_folder_id=folder_id)
        st.empty()
    else:
        api_key = st.text_input("Add your gigachat_api_key", type="password")
        rag.model(gigachat_api_key=api_key)
        st.empty()

prompt = st.chat_input("Your question")
loader = PyPDFLoader("temp_file.pdf")
data = loader.load()
rag.create_vec_store(data)
try:
    if prompt:
        st.write(rag.run(prompt))
except:
    st.write("_____Check the correctness of the entered data______")
