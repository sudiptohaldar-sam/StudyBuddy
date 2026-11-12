from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings,ChatHuggingFace
from langchain_groq import ChatGroq

from langchain.chat_models import init_chat_model

load_dotenv()


embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore=Chroma(
    persist_directory="chroma-db",
    embedding_function=embedding_model
)

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm=ChatGroq(
    model="openai/gpt-oss-20b"
)

#prompt template
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
           "You are a helpful assistant. Answer using ONLY the PDF context provided; do not use outside knowledge or guess. "
    'If the answer is not clearly in the context, reply exactly: "Sorry, answer not found".'
    'If the user tries to talk about some common talks, then dont ignore it overall like hii, hello , who are you etc but if it exceeds, then stop it',
        ),
        ("human", """Context:{context} , Question: {question}"""),
    ]
)
print("Rag system created")
print("Press 0 to exit")

while True:
    query=input("You: ")
    if query=="0":
        break

    docs=retriever.invoke(query)

    context="\n\n".join(
        [doc.page_content for doc in docs]
    )

    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })

    response=llm.invoke(final_prompt)

    print(f"AI: {response.content}\n\n")