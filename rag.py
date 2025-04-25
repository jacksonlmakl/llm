import os
import sys
import pandas as pd
from PIL import Image
import pytesseract
import docx2txt
import PyPDF2
import csv
import openpyxl
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from transformers import pipeline
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, directory_path):
        logging.disable(logging.CRITICAL)  # Disable all logging
        self.directory_path = directory_path
        # Initialize the embedding model
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize document storage
        self.documents = []
        
        # Initialize Tesseract for OCR (make sure tesseract is installed)
        pytesseract.pytesseract.tesseract_cmd = r'tesseract'  # Update this path if needed
        
    def process_directory(self):
        """Process all files in the directory and extract text"""
        logger.info(f"Processing directory: {self.directory_path}")
        
        if not os.path.exists(self.directory_path):
            logger.error(f"Directory not found: {self.directory_path}")
            return False
            
        file_count = 0
        
        for filename in tqdm(os.listdir(self.directory_path)):
            filepath = os.path.join(self.directory_path, filename)
            if os.path.isfile(filepath):
                try:
                    file_extension = os.path.splitext(filename)[1].lower()
                    text = self.extract_text(filepath, file_extension)
                    if text:
                        self.documents.append(Document(page_content=text, metadata={"source": filename}))
                        file_count += 1
                except Exception as e:
                    logger.error(f"Error processing {filename}: {str(e)}")
        
        logger.info(f"Successfully processed {file_count} files")
        return True
    
    def extract_text(self, filepath, file_extension):
        """Extract text from different file types"""
        logger.debug(f"Extracting text from {filepath} with extension {file_extension}")
        
        if file_extension in ['.txt']:
            return self.extract_from_txt(filepath)
        elif file_extension in ['.pdf']:
            return self.extract_from_pdf(filepath)
        elif file_extension in ['.docx', '.doc']:
            return self.extract_from_docx(filepath)
        elif file_extension in ['.csv']:
            return self.extract_from_csv(filepath)
        elif file_extension in ['.xlsx', '.xls']:
            return self.extract_from_xlsx(filepath)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            return self.extract_from_image(filepath)
        else:
            logger.warning(f"Unsupported file type: {file_extension}")
            return None
    
    def extract_from_txt(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                return file.read()
        except Exception as e:
            logger.error(f"Error extracting text from {filepath}: {str(e)}")
            return ""
    
    def extract_from_pdf(self, filepath):
        try:
            text = ""
            with open(filepath, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {filepath}: {str(e)}")
            return ""
    
    def extract_from_docx(self, filepath):
        try:
            return docx2txt.process(filepath)
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {filepath}: {str(e)}")
            return ""
    
    def extract_from_csv(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                csv_reader = csv.reader(file)
                rows = list(csv_reader)
                
                # Convert CSV to a readable text format
                headers = rows[0] if rows else []
                text = "CSV File Content:\n"
                
                # Add headers
                text += "Headers: " + ", ".join(headers) + "\n\n"
                
                # Add data rows
                for i, row in enumerate(rows[1:], start=1):
                    row_text = f"Row {i}: "
                    for j, cell in enumerate(row):
                        if j < len(headers):
                            row_text += f"{headers[j]}: {cell}, "
                    text += row_text + "\n"
                    
                return text
        except Exception as e:
            logger.error(f"Error extracting text from CSV {filepath}: {str(e)}")
            return ""
    
    def extract_from_xlsx(self, filepath):
        try:
            workbook = openpyxl.load_workbook(filepath, data_only=True)
            text = f"Excel File: {os.path.basename(filepath)}\n\n"
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text += f"Sheet: {sheet_name}\n"
                
                # Get all values from the sheet
                data = []
                for row in sheet.iter_rows(values_only=True):
                    data.append(row)
                
                if not data:
                    text += "Empty sheet\n\n"
                    continue
                    
                # Extract headers (first row)
                headers = [str(cell) if cell is not None else "" for cell in data[0]]
                
                # Process data rows
                for i, row in enumerate(data[1:], start=2):
                    row_text = f"Row {i-1}: "
                    for j, cell in enumerate(row):
                        if j < len(headers) and headers[j]:
                            row_text += f"{headers[j]}: {str(cell) if cell is not None else 'N/A'}, "
                    text += row_text + "\n"
                
                text += "\n"
            
            return text
        except Exception as e:
            logger.error(f"Error extracting text from Excel {filepath}: {str(e)}")
            return ""
    
    def extract_from_image(self, filepath):
        try:
            img = Image.open(filepath)
            text = pytesseract.image_to_string(img)
            if not text:
                return f"Image file without extractable text: {os.path.basename(filepath)}"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from image {filepath}: {str(e)}")
            return ""
    
    def create_vector_store(self, save_path=None):
        """Create a vector store from chunked documents"""
        if not self.documents:
            logger.warning("No documents to process.")
            return None
            
        logger.info("Chunking documents...")
        chunked_docs = []
        
        for doc in tqdm(self.documents):
            chunks = self.text_splitter.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                metadata = doc.metadata.copy()
                metadata["chunk"] = i
                chunked_docs.append(Document(page_content=chunk, metadata=metadata))
        
        logger.info(f"Created {len(chunked_docs)} chunks from {len(self.documents)} documents")
        
        if not chunked_docs:
            logger.warning("No chunks created.")
            return None
            
        logger.info("Creating vector store...")
        vector_store = FAISS.from_documents(chunked_docs, self.embeddings)
        
        # Save the vector store if a path is provided
        if save_path:
            logger.info(f"Saving vector store to {save_path}")
            vector_store.save_local(save_path)
        
        return vector_store

class RAGSystem:
    def __init__(self, vector_store=None, vector_store_path=None):
        """Initialize the RAG system"""
        if vector_store:
            self.vector_store = vector_store
        elif vector_store_path:
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            self.vector_store = FAISS.load_local(vector_store_path, self.embeddings)
        else:
            raise ValueError("Either vector_store or vector_store_path must be provided")
    
    def query(self, prompt, k=5):
        """
        Query the vector store and return the prompt enriched with relevant information
        
        Args:
            prompt: The user's prompt/question
            k: Number of relevant chunks to retrieve
            
        Returns:
            Enriched prompt with relevant context
        """
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(prompt, k=k)
        
        # Extract relevant context
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Create sources list for citation
        sources = []
        for doc in retrieved_docs:
            source = doc.metadata.get("source", "Unknown")
            if source not in sources:
                sources.append(source)
        
        # Format sources
        sources_text = "\n".join([f"- {source}" for source in sources])
        
        # Combine context with the prompt for a better response
        enriched_prompt = f"""
        Question/Prompt: {prompt}
        
        Here is relevant information that may help answer the question:
        
        {context}
        
        Sources consulted:
        {sources_text}
        
        Based on the above information, here's a comprehensive answer to the original question/prompt:
        """
                
        return enriched_prompt

        
def RAG(prompt,directory_path = "documents",vector_store_path = "vector_store"):
    # Process documents and create vector store
    processor = DocumentProcessor(directory_path)
    processor.process_directory()
    vector_store = processor.create_vector_store(save_path=vector_store_path)
    
    # Create RAG system
    rag_system = RAGSystem(vector_store=vector_store)
    
    # Example query function
    def query_rag(prompt):
        """Query the RAG system with a prompt"""
        return rag_system.query(prompt).replace("\n","").strip()
    
    # Test with a query
    # prompt = "tell me about airbnb?"
    enriched_prompt = query_rag(prompt)
    return enriched_prompt