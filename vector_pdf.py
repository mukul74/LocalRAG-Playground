from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pymupdf  # Using pymupdf instead of fitz

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    return text

# Read PDF files from a directory
pdf_directory = "./pdf_documents"
pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith(".pdf")]

embeddings = OllamaEmbeddings(model="mxbai-embed-large")

db_location = "./chrome_langchain_db_pdf"
add_documents = not os.path.exists(db_location)

documents = []
ids = []

if add_documents:
    for i, pdf_file in enumerate(pdf_files):
        text = extract_text_from_pdf(pdf_file)
        document = Document(
            page_content=text,
            metadata={"source": pdf_file},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)

vector_store = Chroma(
    collection_name="pdf_documents",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(search_kwargs={"k": 5})