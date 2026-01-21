import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from docling.document_converter import DocumentConverter
from langchain_core.documents import Document
import uuid
from openai import OpenAI
class FinancialTableProcessor:
    def __init__(self):
        self.converter = get_gpu_converter() # Initializes the docling engine. It ensures that ML models are loaded into memory
    # This function is called when the process_pdf function detects a table that can later be converted to markdown-KV.
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def _summarize_table(self, table_markdown: str) -> str:
        """
        Uses an LLM to generate a dense semantic summary of the table.
        This summary becomes the 'Context' for retrieval.
        """
        prompt = f"""
        You are a financial analyst. Analyze the following balance sheet table.
        Output a single sentence summary describing EXACTLY what this table contains.
        Include the financial year, the nature of items (e.g., "Non-current investments"), and any specific notes mentioned.

        Table Markdown:
        {table_markdown[:3000]} # Truncate to save tokens if table is massive

        Summary:
        """
        response = self.client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system",
             "content": "You are a helpful assistant that summarize tables in one line which contains all the important information about it."
            },
            {"role": "user",
             "content": prompt
            }
        ]
    )
        return response.choices[0].message.content

    def _dataframe_to_markdown_kv(self, df: pd.DataFrame, context_header: str) -> List[str]:
        """
        Transforms a DataFrame into a list of Markdown-KV strings.
        Each string represents ONE row but carries the semantic weight of the headers.
        """
        """
        Transforms a DataFrame into a list of Markdown-KV strings.
        Each string represents ONE row but carries the semantic weight of the headers.
        """
        kv_chunks = []
        # Clean headers to be string and strip whitespace
        headers = [str(h).strip() for h in df.columns] # Creates a list of headers that consist of column names which will later be
                                                       # used to create the Key: Value pairs

        for index, row in df.iterrows(): # Iterates over each and every row of the pandas dataframe
            # Start the chunk with the Global Context (e.g., "Note 4: Long Term Borrowings")
            chunk_parts = [f"## Context: {context_header}"] # Gives the context so that LLM can understand from which table the data
                                                            # came from
                                                            # Temprary list building text for current row
            # Iterate through cells to create Key: Value pairs
            for col_idx, value in enumerate(row):
                if col_idx < len(headers):
                    key = headers[col_idx] # Grabs the column name corresponding to the cell
                    val = str(value).strip() # Grabs the value while converting them to string

                    # Skip empty cells to reduce noise (common in financial tables)
                    # val is nan(not a number) or it is empty then dont append it to the table
                    if val and val.lower()!= "nan" and val!= "":
                        chunk_parts.append(f"- **{key}**: {val}")

            # Only add if the row has meaningful data (more than just the header)
            if len(chunk_parts) > 1:
                kv_chunks.append("\n".join(chunk_parts))

        return kv_chunks

    def process_pdf(self, pdf_path: str) -> Dict: # This method orchestrates the file processing and creates the Parent-Child structure
        """
        Parses PDF, extracts tables, and creates Parent-Child relationships.
        Input: Path to the uploaded PDF file.
        """
        result = self.converter.convert(pdf_path)# Docling scans the PDF, identifies table boundaries, performs OCR, and reconstructs the layout.
        doc = result.document # Holds the structured representation of the PDF (headings, paragraphs, tables).

        # We initialize two lists: one for the Parents and one for the "Searchable Details" (Children).
        parents = []
        children = []

        # 1. Process Tables (The "Hard" Data)
        for i, table in enumerate(doc.tables):
            # Export full table as Markdown for the PARENT (Holistic View)
            table_md = table.export_to_markdown()
            parent_id = str(uuid.uuid4())
            # 2. GENERATE DYNAMIC CONTEXT
            # Instead of static text, we ask the LLM "What is this table?"
            # Result: "Note 14: Breakdown of Deferred Tax Liabilities for FY 2023-24"
            try:
                table_summary = self._summarize_table(table_md)
                print(f"Table {i+1} Summary: {table_summary}")
            except Exception as e:
                print(f"Summarization failed for table {i}, using fallback.")
                table_summary = "Financial Statement Table"

            # Create the parent
            parent_doc = Document(
                page_content=f"Summary: {table_summary}\n\n{table_md}",
                metadata= {"source": pdf_path,
                          "type": "table_parent",
                          "explicit_id": parent_id, # We will link this to the child ID in the VectorStoreManager
                          }
            )
            parents.append(parent_doc)

            # Create Children (Rows transformed to Markdown-KV)
            df = table.export_to_dataframe()
            kv_rows = self._dataframe_to_markdown_kv(df, table_summary)

            for row_text in kv_rows:
                child_doc = Document(
                    page_content=row_text,
                    metadata={
                        "source": pdf_path,
                        "type": "table_row_child",
                        "parent_id": parent_id # Link to Parent
                    }
                )
                children.append(child_doc)

        return {"parents": parents, "children": children}