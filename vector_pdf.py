from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
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

# Initialize embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Vector DB location
db_location = "./chrome_langchain_db_pdf"
add_documents = not os.path.exists(db_location)

documents = []
ids = []

# Initialize LangChain's text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=5,
    separators=["\n\n", "\n", " ", ""]
)

if add_documents:
    for i, pdf_file in enumerate(pdf_files):
        text = extract_text_from_pdf(pdf_file)

        # Split text into chunks
        chunks = text_splitter.split_text(text)

        # Create Document instances for each chunk
        for j, chunk in enumerate(chunks):
            doc_id = f"{i}-{j}"
            document = Document(
                page_content=chunk,
                metadata={"source": pdf_file, "chunk": j},
                id=doc_id
            )
            ids.append(doc_id)
            documents.append(document)

# Initialize Chroma vector store
vector_store = Chroma(
    collection_name="pdf_documents",
    persist_directory=db_location,
    embedding_function=embeddings
)

# Add documents only if needed
if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

# Create retriever
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
