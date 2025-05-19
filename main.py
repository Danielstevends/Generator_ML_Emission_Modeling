import os
import pandas as pd
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

# Disable Hugging Face tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load environment variables
load_dotenv("api_key.env")


def load_datasets(file_paths):
    all_documents = []
    for file_path in file_paths:
        filename = os.path.basename(file_path)
        year = "".join(filter(str.isdigit, filename))  # Extract numeric year from filename

        df = pd.read_csv(file_path)
        data = df.to_string(index=False)

        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = text_splitter.split_text(data)

        all_documents.extend([Document(page_content=doc, metadata={"year": year}) for doc in docs])

    return all_documents


def create_vector_store(documents):
    embeddings = HuggingFaceEmbeddings()
    vector_db = Chroma.from_documents(documents, embeddings)
    return vector_db


def retrieve_relevant_data(vector_db, query, year=None):
    docs = vector_db.similarity_search(query, k=3)

    if year:
        docs = [doc for doc in docs if doc.metadata.get("year") == str(year)]

    return docs


def main():
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("🚨 GROQ_API_KEY is missing! Set it in the environment variables or .env file.")

    model = "llama3-8b-8192"
    groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model)

    print("✅ Loading datasets and preparing knowledge base...")

    dataset_paths = ["results/data_2022.csv", "results/data_2023.csv", "results/data_2024.csv"]

    if not all(os.path.exists(path) for path in dataset_paths):
        raise FileNotFoundError("🚨 One or more dataset files are missing! Check your file paths.")

    documents = load_datasets(dataset_paths)
    vector_db = create_vector_store(documents)

    print("🤖 Hello! I'm your knowledge assistant! I can answer questions using data from 2022, 2023, or 2024.")

    system_prompt = "You are a knowledge assistant that retrieves relevant information before answering."
    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)

    while True:
        user_question = input("🔹 Ask a question (or type 'exit' to quit): ")
        if user_question.lower() == "exit":
            print("👋 Goodbye!")
            break

        year_filter = input("📅 Do you want to filter by year? (Enter year or 'no'): ")
        year_filter = year_filter.strip() if year_filter.lower() != "no" else None

        retrieved_docs = retrieve_relevant_data(vector_db, user_question, year_filter)
        retrieved_text = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessagePromptTemplate.from_template(
                f"📚 Context: {retrieved_text}\n\n👤 User: {{human_input}}"
            ),
        ])


        # ✅ Extract chat history correctly as a list
        chat_history = memory.load_memory_variables({}).get("chat_history", [])

        # ✅ New RunnableSequence method (Replaces LLMChain)
        chain = prompt | groq_chat
        response = chain.invoke({"chat_history": chat_history, "human_input": user_question})

        print(f"🤖 Chatbot ({year_filter if year_filter else 'All Years'}):", response)


if __name__ == "__main__":
    main()
