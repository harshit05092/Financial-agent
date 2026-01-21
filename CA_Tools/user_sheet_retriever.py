import os
import uuid
import pandas as pd
from typing import List, Dict, Any

# --- 1. SAFE IMPORTS ---
try:
    from docling.document_converter import DocumentConverter
    from langchain_core.documents import Document
    from openai import OpenAI
    from langchain_openai import OpenAIEmbeddings
    import chromadb
    # Import class explicitly to avoid module confusion
    from langchain_chroma import Chroma
    from langchain_classic.storage._lc_store import create_kv_docstore
    from langchain_classic.storage import LocalFileStore
    from langchain_core.stores import InMemoryByteStore
    from langchain_classic.retrievers import ParentDocumentRetriever
    from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
except ImportError as e:
    raise ImportError(f"Missing dependency in my_utils.py: {e}")
# from unstructured.embed.openai import OpenAIEmbeddingConfig, OpenAIEmbeddingEncoder
class CAFinancialAgent: # Blueprint for the agent manager
    def __init__(self,
                 OPENAI_API_KEY : str,
                 local_working_dir="/content/tax_knowledge_base", # Path where the tax docs will be stored
                 ):
        self.OPENAI_API_KEY = OPENAI_API_KEY
        self.embedding_fn = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        # self.local_working_dir = local_working_dir

        # --- PARTITION 1: PERMANENT STORAGE (Income Tax Act) ---
        # Initialize Persistent Client
        # print("Setting up persistent client")
        # self.persistent_client = chromadb.PersistentClient(path=os.path.join(local_working_dir, "chroma_db"))

        # print("Persistent client made. Now making collection")
        # self.tax_collection = self.persistent_client.get_or_create_collection(
        #     name="indian_income_tax_act"
        # )# Checks if there is a collection named indian_income_tax_act. If not then creates one
        # print("Collection made. Now making vectorstore")
        # self.tax_vectorstore = Chroma(
        #     client=self.persistent_client,
        #     collection_name="indian_income_tax_act",
        #     embedding_function=self.embedding_fn
        # )# Wraps raw chroma client into compatible vector store object. Will be used later for retrival

        # File store for Parent Documents (Full Sections)
        # fs_path = os.path.join(local_working_dir, "docstore")
        # if not os.path.exists(fs_path): os.makedirs(fs_path)
        # print("Docstore made. Now making docstore")
        # self.tax_docstore = create_kv_docstore(LocalFileStore(fs_path))# LocalFileStore creates a storage system on the hard drive to
                                                                       # store file. create_kv_docstore converts to key-value pair.
                                                                       # Looks like dictionary to langchain.
        # --- PARTITION 2: EPHEMERAL STORAGE (User Data) ---
        # Will be initialized per session
        self.user_retriever = None # Placeholder for user data. Empty at beginning for the user to upload file.
        self.user_docstore = None # In-Memory

    # ---------------------------------------------------------
    # ACTION A: Ingest Tax Act (Run once, stored forever)
    # ---------------------------------------------------------
    # def ingest_tax_act(self, folder_path_with_pdfs: str):
    #     """
    #     Reads 15+ PDFs (Sections) and stores them permanently.
    #     Strategy: File = Parent, Sub-sections = Children.
    #     """
    #     # Quick check if already indexed to save money/time
    #     if self.tax_collection.count() > 0: # If the database already has files then it will startup quickly without taking time
    #         print(f"✅ Tax DB already contains {self.tax_collection.count()} chunks. Skipping ingestion.")
    #         return

    #     print("⚠️ Indexing Income Tax Act... This may take time.")

    #     # Initializes IndianLegalHierarchySplitte and docling
    #     splitter = IndianLegalHierarchySplitter()
    #     converter = get_gpu_converter() # Converts PDF to Markdown

    #     pdf_files = [f for f in os.listdir(folder_path_with_pdfs) if f.endswith('.pdf')] # Finds all pdf files in the folder

    #     parents_to_add = [] # Full file
    #     children_to_add = [] # Sub-section

    #     for pdf_file in pdf_files:
    #         full_path = os.path.join(folder_path_with_pdfs, pdf_file)
    #         print(f"Processing Section: {pdf_file}")

    #         # 1. Parse PDF
    #         result = converter.convert(full_path) # Reads pdf file using AI vision
    #         full_text = result.document.export_to_markdown() # Converts to markdown format

    #         # 2. Create Parent (The Full Section)
    #         parent_id = str(uuid.uuid4()) # Assigns unique id to parent
    #         parent_doc = Document(page_content=full_text, metadata={"source": pdf_file}) # Converts to langchain document content
    #         parents_to_add.append((parent_id, parent_doc)) # Stores both parent id and document

    #         # 3. Create Children (Sub-sections)
    #         chunks = splitter.split_text(full_text) # Uses IndianLegalHierarchySplitter() for splitting text
    #         for chunk in chunks:
    #             if len(chunk) > 20: # Filter noise
    #                 child_doc = Document(
    #                     page_content=chunk,
    #                     metadata={
    #                         "source": pdf_file,
    #                         "doc_id": parent_id # Link to Parent
    #                     }
    #                 ) # It stamps the Child with the Parent's ID. When the retriever finds this child later, it will use this ID to fetch the Parent.
    #                 children_to_add.append(child_doc)

    #     # 4. Bulk Save
    #     if children_to_add:
    #         print(f"🧠 Embedding {len(children_to_add)} child chunks into VectorStore...")
    #         self.tax_vectorstore.add_documents(children_to_add)
    #         print("✅ Tax Act Indexing Complete.")

    #         # Save backup if running on Colab (Optional but recommended)
    #         if hasattr(self, 'save_to_drive'):
    #             self.save_to_drive()
    #     else:
    #         print("⚠️ Warning: No valid child chunks were extracted. The database is empty.")

    #     if parents_to_add:
    #         self.tax_docstore.mset(parents_to_add)
    #         self.tax_vectorstore.add_documents(children_to_add)
    #         print("✅ Tax Act Indexing Complete.")


    # ---------------------------------------------------------
    # ACTION B: Upload User Balance Sheet (Run per session)
    # ---------------------------------------------------------
    def upload_user_balance_sheet(self, file_path : str) -> ParentDocumentRetriever: # This method runs every time a user uploads a file. It creates a RAM-only database.
        """
        Ingests user PDF into RAM (Ephemeral).
        Strategy: Table = Parent, Markdown-KV Row = Child.
        """

        parent_id = str(uuid.uuid4())
        print(f"🔄 Processing User Document: {file_path}")

        # 1. Initialize Ephemeral Infrastructure
        # We use a purely in-memory Chroma client for privacy
        ephemeral_client = chromadb.EphemeralClient()

        user_vectorstore = Chroma(
            client=ephemeral_client,
            collection_name="current_session_financials",
            embedding_function=self.embedding_fn
        )# Creates a langchain object that can be used for retrieval later

        # In-Memory Docstore for Parent Tables
        self.user_docstore = InMemoryByteStore()

        # 2. Process PDF
        processor = FinancialTableProcessor(self.OPENAI_API_KEY)
        data = processor.process_pdf(file_path)

        # 3. Store Parents (Full Tables)
        # 'parents' is a list of Docs. We need (id, Doc) tuples for mset.
        # We stored the generated ID in metadata["explicit_id"] earlier.
        # docstore.mset() command (which saves data to memory/disk) expects a list of Tuples in the format (ID, Document).
        # It does not accept just a list of Documents.
        parent_pairs = [(p.metadata["explicit_id"], p) for p in data["parents"]]
        self.user_docstore.mset(parent_pairs)

        # 4. Store Children (Rows)
        if data["children"]:
            user_vectorstore.add_documents(data["children"])

        child_splitter_dummy = RecursiveCharacterTextSplitter(chunk_size=400)

        # 5. Create Retriever
        self.user_retriever = ParentDocumentRetriever(
            vectorstore=user_vectorstore,
            docstore=self.user_docstore,
            child_splitter=child_splitter_dummy, # Pass the dummy splitter here
            parent_splitter=None, # This is optional, can be None
            id_key="doc_id" # This ensures it looks for 'doc_id' in the child metadata
        )
        print("✅ User Balance Sheet loaded into secure memory.")

    # ---------------------------------------------------------
    # ACTION C: Get Tools for Agents
    # ---------------------------------------------------------
    def get_agent_tools(self): # Will help plugging these retrievers into langGraph
        """
        Returns the two specific retrievers needed by your LangGraph Supervisor.
        """
        # child_splitter_dummy = MockSplitter()
        # parent_splitter_dummy = MockSplitter()
        # 1. Tax Retriever (Permanent)
        # tax_retriever = ParentDocumentRetriever(
        #     vectorstore=self.tax_vectorstore,
        #     docstore=self.tax_docstore,
        #     child_splitter=child_splitter_dummy,
        #     parent_splitter=parent_splitter_dummy,
        #     id_key="doc_id",  # Mandatory: Links child chunks to parent docs
        #     search_kwargs={"k": 5} # Retrieve top 5 relevant contexts
        # )

        # 2. User Retriever (Ephemeral) - check if session is active
        if not self.user_retriever:
            raise ValueError("No user session active. Please upload a balance sheet first.")

        return self.user_retriever