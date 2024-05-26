from langchain.document_loaders import PyMuPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS



def load_pdf(data_folder):
    loader = DirectoryLoader(data_folder, glob="*.pdf", loader_cls=PyMuPDFLoader)
    documents = loader.load()
    text_documents = [doc for doc in documents if doc.page_content]
    return text_documents
def text_split(text_documents):
    text_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    for doc in text_documents:
        chunks = text_splitter.split_documents([doc])
        text_chunks.extend(chunks)
    return text_chunks
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings

def create_faiss_index(text_chunks, embeddings):
    """
    Create a FAISS index from the text chunks using the specified embeddings.
    
    Parameters:
    text_chunks (List[TextChunk]): List of text chunks.
    embeddings (HuggingFaceEmbeddings): The embeddings model.
    
    Returns:
    FAISS: The created FAISS index.
    """
    texts = [t.page_content for t in text_chunks]
    faiss_index = FAISS.from_texts(texts, embeddings)
    return faiss_index
