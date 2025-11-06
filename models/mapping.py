from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ==================== Pydantic Models ====================

class ItemMapping(BaseModel):
    item_code: Optional[str] = Field(None, description="Unique code for the invoice item, if available")
    item_description_invoice: str = Field(..., description="Item description as per invoice")
    item_description_po: Optional[str] = Field(None, description="Matched PO item description (if any)")
    mapping_status: Literal["MAPPED", "UNMAPPED"] = Field(...,
                                                          description="Whether the invoice item was matched to a PO item")
    remarks: Optional[str] = Field(None, description="Short note about the match quality or reason")


class MappingSummary(BaseModel):
    total_invoice_items: int = Field(0, description="Total number of items in the invoice")
    total_mapped_items: int = Field(0, description="Number of matched items")
    total_unmapped_items: int = Field(0, description="Number of unmatched items")


class MappingReport(BaseModel):
    invoice_number: Optional[str] = None
    po_number: Optional[str] = None
    grn_number: Optional[str] = None
    vendor_name: Optional[str] = None
    items: List[ItemMapping] = Field(default_factory=list)
    summary: Optional[MappingSummary] = None