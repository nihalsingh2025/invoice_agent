import os,sys
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import cv2
import json
import sqlite3
import numpy as np
import pandas as pd
from openai import OpenAI, APIError
import tempfile
from pdf2image import convert_from_path
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from oracle_connector.fetch_po_grn_data import DataFetcherOraclePOGRN
load_dotenv()

dsn = os.getenv('ORACLE_DSN')
username = os.getenv('ORACLE_USERNAME')
password = os.getenv('ORACLE_PASSWORD')
api_key = os.getenv("OPENAI_API_KEY")

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

class ItemMapping(BaseModel):
    """Represents simple mapping info for a single invoice item."""
    item_code: Optional[str] = Field(None, description="Unique code for the invoice item, if available")
    item_description_invoice: str = Field(..., description="Item description as per invoice")
    item_description_po: Optional[str] = Field(None, description="Matched PO item description (if any)")
    mapping_status: Literal["MAPPED", "UNMAPPED"] = Field(..., description="Whether the invoice item was matched to a PO item")
    remarks: Optional[str] = Field(None, description="Short note about the match quality or reason")

class MappingSummary(BaseModel):
    """Overall summary of mapping results."""
    total_invoice_items: int = Field(0, description="Total number of items in the invoice")
    total_mapped_items: int = Field(0, description="Number of matched items")
    total_unmapped_items: int = Field(0, description="Number of unmatched items")

class MappingReport(BaseModel):
    """Full report combining item-level mappings and summary."""
    invoice_number: Optional[str] = None
    po_number: Optional[str] = None
    grn_number: Optional[str] = None
    vendor_name: Optional[str] = None
    items: List[ItemMapping] = Field(default_factory=list)
    summary: Optional[MappingSummary] = None

llm = ChatOpenAI(model="gpt-4.1", temperature=0, openai_api_key=api_key)

def extract_text_with_ocr(pdf_path:str)->str:

    pdf_name = pdf_path.split("/")[1].split(".")[0]
    output_txt = f" {pdf_name}.txt"

    images = convert_from_path(pdf_path, dpi=200)

    print("pdf converted to img")
    model = ocr_predictor(pretrained=True,assume_straight_pages=False,straighten_pages=True)

    all_text = ""

    with open(output_txt, "w", encoding="utf-8") as f:
        for page_num, pil_image in enumerate(images):

            # plt.figure(figsize=(10, 12))
            # plt.imshow(pil_image)
            # plt.axis('off')
            # plt.title(f"Page {page_num + 1} - RGB")
            # plt.show()

            # Convert PIL to OpenCV

            cv_image = np.array(pil_image)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            print(f"page {page_num} img converted to grayscale")

            # plt.figure(figsize=(10, 12))
            # plt.imshow(gray, cmap='gray')
            # plt.axis('off')
            # plt.title(f"Page {page_num + 1} - Grayscale")
            # plt.show()


            # Save temporarily as a PNG image

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, gray)

            doc = DocumentFile.from_images(tmp_path)

            print("page extraction started")
            result = model(doc)
            page_data = result.export()['pages'][0]

            page_header = f"\n===== PAGE {page_num + 1} =====\n"
            f.write(page_header)
            all_text += page_header
            f.write(page_header)


            for block in page_data['blocks']:
                for line in block['lines']:
                    text_line = " ".join([word['value'] for word in line['words']])
                    f.write(text_line + "\n")
                    all_text += text_line + "\n"

    print(f"\n OCR extraction completed. Text saved to: {output_txt}")

    return all_text

def structure_info_with_llm(ocr_text:str)->InvoiceSchema:

    llm_with_schema = llm.with_structured_output(InvoiceSchema)

    prompt = f"""
    You are an intelligent data extraction assistant specializing in reading messy OCR text from scanned invoices.

    Your task:
    - Read the raw OCR text carefully. It may be misaligned, incomplete, or have overlapping text.
    - Identify and extract key invoice fields accurately.
    - For each field, return only the most relevant and contextually correct value.
    - If a field cannot be confidently identified, set it to "Not provided" or 0.0 as appropriate."

    When identifying the PO Number:
    - Look specifically for terms such as "PO No", "P.O. No", "Purchase Order No", "Your Ref/LPO", "L.P.O No", or "Customer LPO No".
    When identifying the Release Number:
    - It often appears as a suffix or extension to the PO number, separated by a dash or letter.

    For example:    
    24004602-2 → Release number is 2
    24004593 release 1 → Release number is 1
    24004078 R7 → Release number is 7
    If no explicit release number is found, default to 1.
    OCR Text:
    {ocr_text}
    """

    print("llm structuring the text...")

    invoice_details = llm_with_schema.invoke(prompt)

    print("############################ INVOICE Details ############################")

    print(invoice_details)

    return invoice_details

def fetch_po_grn_details(po_number:int)->dict:

    fetcher = DataFetcherOraclePOGRN(dsn=dsn, user_name=username, psswrd=password)
    print("database_connected")

    fetcher.connect()

    po_details = fetcher.fetch_po_data(po_number)

    grn_details = fetcher.fetch_grn_data(po_number)

    fetcher.disconnect()

    print("############################## PO Details #############################")
    print(po_details)

    print("############################# GRN Details #############################")
    print(grn_details)

    return {"po_details": po_details, "grn_details": grn_details}

def filter_po_data_on_release(data_dict: dict, release_num: str)->pd.DataFrame:
    """
    Filters PO and GRN data based on release number.
    """
    print("filtering data on release...")
    # Convert JSON to DataFrames
    po_df = pd.DataFrame(json.loads(data_dict.get("po_details")))

    release_col = po_df["RELEASE_NUM"].dropna().unique().tolist()
    if len(release_col) == 0:
        print("no release number found")
        return po_df

    else:
        po_df = po_df[po_df["RELEASE_NUM"].astype(str) == str(release_num)]
        print(f"After filtering po:{po_df.shape} ")
        return po_df

def get_text_embedding(text: str) -> list:
    """Get OpenAI embedding for a given text."""
    try:
        client = OpenAI(api_key=api_key)
        response = client.embeddings.create(
            input=text,
            model="text-embedding-3-large"
        )
        return response.data[0].embedding
    except APIError as e:
        print(f"Embedding API error: {e}")
        return []

def filter_by_desc_similarity(invoice_item_descriptions: list, df: pd.DataFrame, desc_col: str = "ITEM_DESCRIPTION", threshold: float = 0.5) -> pd.DataFrame:
    """
    Filters a DataFrame based on cosine similarity using OpenAI embeddings.
    Keeps rows whose description matches any invoice item description >= threshold.
    """

    if df.empty:
        print("PO DataFrame is empty")
        return df

    if not invoice_item_descriptions:
        print("No invoice item descriptions provided.")
        return df

    if len(df) < 10:
        print(f"Only {len(df)} rows in DataFrame — skipping similarity filtering.")
        return df

    print("Filtering by semantic similarity...")

    df = df.dropna(subset=[desc_col])
    unique_desc = df[desc_col].drop_duplicates().tolist()

    # Generate embeddings
    print("Generating embeddings... ")
    inv_embeddings = [get_text_embedding(text) for text in invoice_item_descriptions]
    df_embeddings = [get_text_embedding(text) for text in unique_desc]

    # Compare each invoice description with all PO/GRN descriptions
    matched_desc = set()
    for inv_vec in inv_embeddings:
        if inv_vec is None:
            continue
        similarities = cosine_similarity([inv_vec], df_embeddings)[0]
        for desc, sim in zip(unique_desc, similarities):
            if sim >= threshold:
                matched_desc.add(desc)

    print(f"{len(matched_desc)} item descriptions matched by cosine similarity ")

    filtered_df = df[df[desc_col].isin(matched_desc)]
    return filtered_df

def generate_report_with_llm(full_context:str)->MappingReport:

    llm_with_schema = llm.with_structured_output(MappingReport)

    report_prompt = f"""
You are an expert financial data analyst tasked with reconciliation.

Given:
1. Extracted invoice data
2. Purchase Order (PO) data from database
3. Goods Receipt Note (GRN) data from database

Your task:
- Analyze and cross-check the details across the three documents..
- Summarize key findings in a short report.
- Any discrepancies or missing links (e.g., missing GRN or PO).
- Identify mismatches or confirmations between invoice, PO, and GRN.
- A short summary status for the entire transaction.

Respond strictly in the structured format provided by the schema.

### Comparison Requirements

1. **Match Logic**
   - Primary matching should be done using `po_number` or a strong similarity in `item_description`.
   - If item descriptions differ slightly (e.g., “Jerry Can 10L Plain” vs “Plain Jerrycan 10LT”), treat them as the same using your reasoning.
   - While matching PO, GRN, and invoice data, **consider only those entries with the same release number** as in the invoice.

2. **Field Comparison**
   - Compare ordered, received, and invoiced quantities.
   - Compare prices and detect mismatches with fair tolerance.
   - Compute totals (`po_amount`, `grn_amount`, `invoice_amount`).
   - Flag and explain discrepancies accurately.

3. **Status Flags**
   - For overall reconciliation, set: `MATCHED`, `PARTIAL MATCH`, `MISMATCH`, or `UNRECONCILED`.

4. **Natural Reasoning**
   - Treat spelling variations, abbreviations, and word order differences as potentially similar.
   - If unsure, describe your reasoning briefly in `description_similarity_note`.

The given data is:
{full_context}
"""

    print("Generating the report...")

    generated_report = llm_with_schema.invoke(report_prompt)

    print("############################ Generated Report ############################")

    print(generated_report)

    return generated_report

def store_report_in_db(report: MappingReport, db_path="reports/invoice_reports.db") -> None:
    import sqlite3

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create main summary table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mapping_reports (
            invoice_number TEXT PRIMARY KEY,
            po_number TEXT,
            grn_number TEXT,
            vendor_name TEXT,
            total_invoice_items INTEGER,
            total_mapped_items INTEGER,
            total_unmapped_items INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Create item details table (✅ fixed)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mapping_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            invoice_number TEXT,
            item_code TEXT,
            item_description_invoice TEXT,
            item_description_po TEXT,
            mapping_status TEXT,
            remarks TEXT,
            FOREIGN KEY(invoice_number) REFERENCES mapping_reports(invoice_number)
        )
    """)

    # Insert summary record
    summary = report.summary
    cursor.execute("""
        INSERT OR REPLACE INTO mapping_reports (
            invoice_number, po_number, grn_number, vendor_name,
            total_invoice_items, total_mapped_items, total_unmapped_items
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        report.invoice_number,
        report.po_number,
        report.grn_number,
        report.vendor_name,
        summary.total_invoice_items,
        summary.total_mapped_items,
        summary.total_unmapped_items,
    ))

    # Remove existing items for this invoice
    cursor.execute("DELETE FROM mapping_items WHERE invoice_number = ?", (report.invoice_number,))

    # Insert line items
    for item in report.items:
        cursor.execute("""
            INSERT INTO mapping_items (
                invoice_number, item_code, item_description_invoice,
                item_description_po, mapping_status, remarks
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            report.invoice_number,
            item.item_code,
            item.item_description_invoice,
            item.item_description_po,
            item.mapping_status,
            item.remarks
        ))

    conn.commit()
    conn.close()
    print(f"✅ Mapping report for invoice {report.invoice_number} stored in database: {db_path}")

if __name__ == "__main__":

    extracted_text = extract_text_with_ocr("invoice_data/kobian.pdf")
    if extracted_text:
        structured_text=structure_info_with_llm(extracted_text)

        po_number=structured_text.po_number
        release_number=structured_text.release_number.strip()
        item_descriptions = [item.name for item in structured_text.goods_services_details]

        print(f"found {len(item_descriptions)} item in invoice")

        if po_number:
            po_grn_details = fetch_po_grn_details(int(po_number))

            if po_grn_details:
                if release_number:
                    po_df = filter_po_data_on_release(po_grn_details,release_number)
                    filtered_po_df = filter_by_desc_similarity(item_descriptions, po_df)

                    full_context = {
                        "invoice_details": structured_text.model_dump_json(indent=2),
                        "po_details": filtered_po_df.to_json(orient='records',indent=2),
                        "grn_details": po_grn_details["grn_details"]
                    }

                    full_context_str = json.dumps(full_context, indent=2)

                    generated_report = generate_report_with_llm(full_context_str)
                    store_report_in_db(generated_report)
