import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from htmlTemplates import css, user_template, bot_template
# from langchain_community.embeddings import (
#     HuggingFaceInstructEmbeddings,
#     HuggingFaceEmbeddings,
# )
# from langchain.embeddings import HuggingFaceInstructEmbeddings
# from InstructorEmbedding import INSTRUCTOR
# from langchain.vectorstores import FAISS


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    # embeddings = HuggingFaceInstructEmbeddings(
    #     query_instruction="Represent the query for retrieval: "
    # )
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm=ChatOpenAI()
    memory=ConversationBufferMemory(input_key='chat history',memory_key='chat history', return_messages=True)
    conversation_chain=ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# def handle_userinput(user_question):
#     response=st.session_state.conversation({'question': user_question})
#     st.write(response)
def handle_userinput(user_question):
    # Initialize conversation history if it doesn't exist
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Add the user question to the conversation history
    st.session_state.chat_history.append({'user': user_question})

    # Call the conversation function with the updated conversation history
    response = st.session_state.conversation({'chat_history': st.session_state.chat_history})

    st.write(response)



def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation=None
    
    st.header("Chat with multiple PDFs :books:")
    user_question=st.text_input("Ask a question about your documents :")
    if user_question:
        handle_userinput(user_question)

    st.write(user_template.replace("{{MSG}}", "hello robot"), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "hello human"), unsafe_allow_html=True)

    

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)
                # st.write(raw_text)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)
                st.write(text_chunks)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                #create converstaion chain
                st.session_state.conversation= get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
