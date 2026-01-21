from pydantic import BaseModel, Field
from typing import Literal

class Router(BaseModel):
    """Worker classification schema."""
    next: Literal["Tax_Researcher", "Financial_Analyst", "Calculator", "Finish"]
    message: str = Field(description="Message to the user (if finishing) or instructions to the worker.")

class SectorEvidence(BaseModel):
    txn_id: Optional[str] = Field(None, description="Transaction ID or Chunk ID")
    description: str = Field(..., description="Brief description of the transaction")
    amount: float = Field(..., description="Amount invested")

class SectorTotal(BaseModel):
    sector_name: str = Field(..., description="Name of the sector (Agriculture, Gold, Land, etc.)")
    total_amount: float = Field(..., description="Sum of investments in this sector")
    evidence: List[SectorEvidence] = Field(..., description="List of transactions proving this total")

class AnalystReport(BaseModel):
    """Final structured report of investments by sector."""
    sector_breakdown: List[SectorTotal]
    uncategorized_amount: float = Field(0.0, description="Amount that could not be classified")
    notes: str = Field(..., description="Any warnings about missing data or duplicates")