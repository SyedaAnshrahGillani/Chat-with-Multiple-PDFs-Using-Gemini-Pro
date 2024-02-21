
import streamlit as st
from PyPDF2 import PdfReader
from PyPDF2 import PdfFileReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import io
load_dotenv()

genai.configure(api_key=os.getenv("GENAI_API_KEY"))



def get_pdf_text(pdf_docs):
    text =""
    for pdf in pdf_docs:
       pdf_io =  io.BytesIO(pdf)
       pdf_reader = PdfReader(pdf_io)
            
       text += pdf_reader.extractText()
    return text


#def get_pdf_text(pdf_docs):
#    text =""
#    for pdf in pdf_docs:
#       
        
#        pdf_reader = PdfReader(pdf)
#        for page in range(pdf_reader.numPages):
#            text += pdf_reader.getPage(page).extractText()
 #   return text




def get_text_chunks(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store=FAISS.from_texts(text_chunks,embeddings=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template= """
    Answer the question as detailed as possible from the provided context, make sure to provide all the detals,\
        if the answer is not in the provided context just say, "Answer is not availaible in the context", don't provide the wrong answer.\
        Context:\n {context} \n
        Question:\n {question} \n

        Answer:
        """

    model = ChatGoogleGenerativeAI(model="gemini",temperature=0.3)

    prompt= PromptTemplate(template=prompt_template,input_variables=["context","question"])
    
    chain = load_qa_chain(model,chain_type="stuff",prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response =chain(
        {"input_documents":docs, "question":user_question}
        , return_only_outputs=True)
    
    print(response)
    st.write("Reply: ",response["output_text"])

def get_vector_store(text_chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks,embeddings=embeddings)
    vector_store.save_local("faiss_index")


def main():
    st.set_page_config("conGPT")
    st.header("Chat with multiple PDFs")

    user_question = st.text_input("Ask a question from the PDF files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload PDF files")
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done!")

if __name__ == "__main__":
    main()

