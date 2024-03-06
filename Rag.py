from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
import streamlit as st
from langchain_community.llms import GigaChat
from langchain_community.vectorstores import FAISS
from langchain_community.llms import YandexGPT
from langchain.schema import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.embeddings import HuggingFaceEmbeddings

class Rag:
    def __init__(self):
        self.llm = None
        self.db = None
        self.template = """
            Answer the question based only on the following context:
            
            {context}
            
            Question: {question}
            If the answer cannot be given then write 'answer cannot be given'
            The answer should be no more than 70 words.
            """

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=120,
            length_function=len,
        )


    def get_embeddings(self):
        embeddings = HuggingFaceEmbeddings(
            model_name="cointegrated/LaBSE-en-ru", model_kwargs={"device": "cpu"})
        return embeddings
    def model(self, yc_api_key=None, yc_folder_id=None, gigachat_api_key=None) -> None:
        if gigachat_api_key:
            self.llm = GigaChat(credentials=gigachat_api_key, verify_ssl_certs=False)
        if yc_api_key and yc_folder_id:
            self.llm = YandexGPT(api_key=yc_api_key, folder_id=yc_folder_id)

    def create_vec_store(self, data):
        split_documents = self.splitter.create_documents([str(data)])
        self.db = FAISS.from_documents(split_documents, self.get_embeddings())

    def format_docs(self, docs):
        return "\n\n".join(d.page_content for d in docs)

    def run(self, question: str) -> str:
        prompt = ChatPromptTemplate.from_template(self.template)
        retriever = self.db.as_retriever()
        chain = (
                {"context": retriever | self.format_docs, "question": RunnablePassthrough()}
                | prompt
                | self.llm
                | StrOutputParser()
        )
        return chain.invoke(question)
