from pydantic import BaseModel, Field
from typing import List

# ==================== Pydantic Models ====================

class GoodsService(BaseModel):
    name: str = Field(default="Not provided.")
    description: str = Field(default="Not provided.")
    amount: float = Field(default=0.0)


class TaxDetail(BaseModel):
    tax_type: str = Field(default="Not provided.")
    tax_amount: float = Field(default=0.0)


class InvoiceSchema(BaseModel):
    invoice_number: str = Field(default="Not provided.")
    date: str = Field(default="Not provided.")
    cuin: str = Field(default="Not provided.")
    vendor_name: str = Field(default="Not provided.")
    vendor_address: str = Field(default="Not provided.")
    vendor_contact: str = Field(default="Not provided.")
    po_number: str = Field(default="Not provided.")
    release_number: str = Field(default="1")
    delivery_note_number: str = Field(default="Not provided.")
    sub_total: float = Field(default=0.0)
    total_amount: float = Field(default=0.0)
    currency: str = Field(default="KES")
    total_tax_amount: float = Field(default=0.0)
    goods_services_details: List[GoodsService] = []
    tax_details: List[TaxDetail] = []
    tax_id: str = Field(default="Not provided.")
    vat_pin: str = Field(default="Not provided.")
