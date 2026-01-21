import re
from typing import List
from langchain_text_splitters import TextSplitter

class IndianLegalHierarchySplitter(TextSplitter):
    """
    Splits Indian laws by their logical hierarchy:
    Section -> Sub-section (1) -> Clause (a) -> Sub-clause (i)
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Regex to identify the start of a sub-section/clause
        # Matches patterns like "(1)", "(2A)", "(ix)", "(a)" at the start of a line
        # NOTE: We removed '(?m)' from here to fix the "global flags" error
        self.split_pattern = r"^\s*\((?:[0-9]+|[a-z]+|[ivx]+)[A-Z]*\)"

    def split_text(self, text: str) -> List[str]:
        # 1. Split by the pattern but KEEP the delimiter (the numbering)
        # We pass flags=re.MULTILINE here to make the '^' anchor work correctly
        parts = re.split(f"({self.split_pattern})", text, flags=re.MULTILINE)

        chunks = []
        current_chunk = ""

        for part in parts:
            # Check if this part is a numbering delimiter (e.g., "(1)")
            if re.match(self.split_pattern, part, flags=re.MULTILINE):
                # If we have a previous chunk accumulating, save it
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                # Start new chunk with the number
                current_chunk = part
            else:
                # Add the content to the current number
                current_chunk += part

        # Append the final chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks