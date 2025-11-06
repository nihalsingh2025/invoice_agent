import os
import sys
import json
import logging
from datetime import datetime

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import cv2
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/invoice_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

dsn = os.getenv('ORACLE_DSN')
username = os.getenv('ORACLE_USERNAME')
password = os.getenv('ORACLE_PASSWORD')
api_key = os.getenv("OPENAI_API_KEY")


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


# ==================== Main Functions with @tool Decorator ====================

@tool("Extract Text with OCR")
def extract_text_with_ocr(pdf_path: str) -> str:
    """
    Extract text from PDF using OCR.

    Args:
        pdf_path: Path to the PDF file to extract text from

    Returns:
        Extracted text from all pages of the PDF
    """
    try:
        logger.info(f"Starting OCR extraction for: {pdf_path}")
        pdf_name = pdf_path.split("/")[-1].split(".")[0]
        output_txt = f"{pdf_name}.txt"

        images = convert_from_path(pdf_path, dpi=200)
        logger.info(f"PDF converted to {len(images)} images")

        model = ocr_predictor(pretrained=True, assume_straight_pages=False, straighten_pages=True)
        all_text = ""

        with open(output_txt, "w", encoding="utf-8") as f:
            for page_num, pil_image in enumerate(images):
                try:
                    cv_image = np.array(pil_image)
                    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
                    gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                    logger.info(f"Page {page_num + 1} converted to grayscale")

                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                        tmp_path = tmp.name
                        cv2.imwrite(tmp_path, gray)

                    doc = DocumentFile.from_images(tmp_path)
                    logger.info(f"Starting extraction for page {page_num + 1}")

                    result = model(doc)
                    page_data = result.export()['pages'][0]

                    page_header = f"\n===== PAGE {page_num + 1} =====\n"
                    f.write(page_header)
                    all_text += page_header

                    for block in page_data['blocks']:
                        for line in block['lines']:
                            text_line = " ".join([word['value'] for word in line['words']])
                            f.write(text_line + "\n")
                            all_text += text_line + "\n"

                    os.unlink(tmp_path)

                except Exception as e:
                    logger.error(f"Error processing page {page_num + 1}: {str(e)}")
                    continue

        logger.info(f"OCR extraction completed. Text saved to: {output_txt}")
        return all_text

    except Exception as e:
        logger.error(f"OCR extraction failed: {str(e)}", exc_info=True)
        raise


@tool("Structure Invoice Information")
def structure_info_with_llm(ocr_text: str) -> str:
    """
    Structure OCR text into invoice schema using LLM.

    Args:
        ocr_text: Raw OCR text extracted from invoice PDF

    Returns:
        JSON string of structured invoice data with all fields extracted
    """
    try:
        logger.info("Starting LLM structuring of OCR text")
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
        llm_with_schema = llm.with_structured_output(InvoiceSchema)

        prompt = f"""
        You are an intelligent data extraction assistant specializing in reading messy OCR text from scanned invoices.

        Your task:
        - Read the raw OCR text carefully. It may be misaligned, incomplete, or have overlapping text.
        - Identify and extract key invoice fields accurately.
        - For each field, return only the most relevant and contextually correct value.
        - If a field cannot be confidently identified, set it to "Not provided" or 0.0 as appropriate.

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

        invoice_details = llm_with_schema.invoke(prompt)
        logger.info(f"Successfully structured invoice: {invoice_details.invoice_number}")
        logger.info(f"PO Number: {invoice_details.po_number}, Release: {invoice_details.release_number}")
        logger.info(f"Found {len(invoice_details.goods_services_details)} items in invoice")

        return invoice_details.model_dump_json(indent=2)

    except Exception as e:
        logger.error(f"LLM structuring failed: {str(e)}", exc_info=True)
        raise


@tool("Fetch PO and GRN Details")
def fetch_po_grn_details(po_number: str) -> str:
    """
    Fetch PO and GRN details from Oracle database.

    Args:
        po_number: Purchase Order number to fetch details for

    Returns:
        JSON string containing both PO and GRN details from database
    """
    try:
        logger.info(f"Fetching PO/GRN details for PO: {po_number}")
        fetcher = DataFetcherOraclePOGRN(dsn=dsn, user_name=username, psswrd=password)
        fetcher.connect()
        logger.info("Database connected successfully")

        po_details = fetcher.fetch_po_data(po_number)
        grn_details = fetcher.fetch_grn_data(po_number)

        fetcher.disconnect()
        logger.info("Database disconnected")

        result = {"po_details": po_details, "grn_details": grn_details}
        return json.dumps(result, indent=2)

    except Exception as e:
        logger.error(f"Failed to fetch PO/GRN details: {str(e)}", exc_info=True)
        raise


@tool("Filter PO Data on Release")
def filter_po_data_on_release(data_dict: dict, release_num: str) -> pd.DataFrame:
    """
    Filter PO data based on release number.

    Args:
        data_dict: Dictionary containing po_details and grn_details
        release_num: Release number to filter by

    Returns:
        Filtered DataFrame containing only rows matching the release number
    """
    try:
        logger.info(f"Filtering PO data on release number: {release_num}")

        po_details = data_dict.get("po_details")

        if isinstance(po_details, str):
            po_df = pd.DataFrame(json.loads(po_details))
        elif isinstance(po_details, list):
            po_df = pd.DataFrame(po_details)
        else:
            raise ValueError(f"Unexpected type for po_details: {type(po_details)}")

        release_col = po_df["RELEASE_NUM"].dropna().unique().tolist()
        if len(release_col) == 0:
            logger.warning("No release number found in PO data")
            return po_df
        else:
            po_df = po_df[po_df["RELEASE_NUM"].astype(str) == str(release_num)]
            logger.info(f"After filtering: {po_df.shape[0]} rows remaining")
            return po_df

    except Exception as e:
        logger.error(f"Failed to filter PO data: {str(e)}", exc_info=True)
        raise


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
        logger.error(f"Embedding API error: {e}")
        return []


@tool("Filter by Description Similarity")
def filter_by_desc_similarity(invoice_item_descriptions: list,df_json: str,
                              desc_col: str = "ITEM_DESCRIPTION", threshold: float = 0.5) -> str:
    """
    Filter DataFrame based on cosine similarity using OpenAI embeddings.

    Args:
        invoice_item_descriptions: List of item descriptions from invoice
        df: DataFrame to filter (PO data)
        desc_col: Column name containing descriptions
        threshold: Similarity threshold (0.0 to 1.0)

    Returns:
        JSON string of filtered DataFrame
    """
    try:
        df = pd.read_json(df_json, orient='records')
        if df.empty:
            logger.warning("PO DataFrame is empty")
            return df.to_json(orient='records', indent=2)

        if not invoice_item_descriptions:
            logger.warning("No invoice item descriptions provided")
            return df.to_json(orient='records', indent=2)

        if len(df) < 10:
            logger.info(f"Only {len(df)} rows in DataFrame — skipping similarity filtering")
            return df.to_json(orient='records', indent=2)

        logger.info("Starting semantic similarity filtering")

        df = df.dropna(subset=[desc_col])
        unique_desc = df[desc_col].drop_duplicates().tolist()

        logger.info("Generating embeddings...")
        inv_embeddings = [get_text_embedding(text) for text in invoice_item_descriptions]
        df_embeddings = [get_text_embedding(text) for text in unique_desc]

        matched_desc = set()
        for inv_vec in inv_embeddings:
            if not inv_vec:
                continue
            similarities = cosine_similarity([inv_vec], df_embeddings)[0]
            for desc, sim in zip(unique_desc, similarities):
                if sim >= threshold:
                    matched_desc.add(desc)

        logger.info(f"{len(matched_desc)} item descriptions matched by cosine similarity")

        filtered_df = df[df[desc_col].isin(matched_desc)]
        return filtered_df.to_json(orient='records', indent=2)

    except Exception as e:
        logger.error(f"Similarity filtering failed: {str(e)}", exc_info=True)
        raise


@tool("Generate Reconciliation Report")
def generate_report_with_llm(full_context: str) -> str:
    """
    Generate mapping report using LLM.

    Args:
        full_context: JSON string containing invoice, PO, and GRN data

    Returns:
        JSON string of the generated mapping report with item mappings and summary
    """
    try:
        logger.info("Generating mapping report with LLM")
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
        llm_with_schema = llm.with_structured_output(MappingReport)

        report_prompt = f"""
You are an expert financial data analyst tasked with reconciliation.

Given:
1. Extracted invoice data
2. Purchase Order (PO) data from database
3. Goods Receipt Note (GRN) data from database

Your task:
- Analyze and cross-check the details across the three documents.
- Summarize key findings in a short report.
- Any discrepancies or missing links (e.g., missing GRN or PO).
- Identify mismatches or confirmations between invoice, PO, and GRN.
- A short summary status for the entire transaction.

Respond strictly in the structured format provided by the schema.

### Comparison Requirements

1. **Match Logic**
   - Primary matching should be done using `po_number` or a strong similarity in `item_description`.
   - If item descriptions differ slightly (e.g., "Jerry Can 10L Plain" vs "Plain Jerrycan 10LT"), treat them as the same using your reasoning.
   - While matching PO, GRN, and invoice data, **consider only those entries with the same release number** as in the invoice.

2. **Field Comparison**
   - Compare ordered, received, and invoiced quantities.
   - Compare prices and detect mismatches with fair tolerance.
   - Compute totals (`po_amount`, `grn_amount`, `invoice_amount`).
   - Flag and explain discrepancies accurately.

3. **Status Flags**
   - For overall reconciliation, set: `MAPPED`, `PARTIAL MATCH`, `MISMATCH`, or `UNRECONCILED`.

4. **Natural Reasoning**
   - Treat spelling variations, abbreviations, and word order differences as potentially similar.
   - If unsure, describe your reasoning briefly in `description_similarity_note`.

The given data is:
{full_context}
"""

        generated_report = llm_with_schema.invoke(report_prompt)
        logger.info(f"Successfully generated report for invoice: {generated_report.invoice_number}")
        logger.info(f"Total items: {generated_report.summary.total_invoice_items}, "
                    f"Mapped: {generated_report.summary.total_mapped_items}, "
                    f"Unmapped: {generated_report.summary.total_unmapped_items}")

        return generated_report.model_dump_json(indent=2)

    except Exception as e:
        logger.error(f"Report generation failed: {str(e)}", exc_info=True)
        raise


@tool("Store Report in Database")
def store_report_in_db(report_json: str, db_path: str = "reports/invoice_reports.db") -> str:
    """
    Store mapping report in SQLite database.

    Args:
        report_json: JSON string of the mapping report
        db_path: Path to SQLite database file

    Returns:
        Success message with invoice number
    """
    try:
        logger.info(f"Storing report in database: {db_path}")

        # Parse the report
        report_dict = json.loads(report_json)
        report = MappingReport(**report_dict)

        dir_name = os.path.dirname(db_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Create tables
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

        # Insert summary
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

        # Delete existing items for this invoice
        cursor.execute("DELETE FROM mapping_items WHERE invoice_number = ?", (report.invoice_number,))

        # Insert line items
        for item in report.items:
            cursor.execute("""
                           INSERT INTO mapping_items (invoice_number, item_code, item_description_invoice,
                                                      item_description_po, mapping_status, remarks)
                           VALUES (?, ?, ?, ?, ?, ?)
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
        logger.info(f"Successfully stored report for invoice: {report.invoice_number}")

        return f"✅ Successfully stored report for invoice {report.invoice_number} in {db_path}"

    except Exception as e:
        logger.error(f"Failed to store report in database: {str(e)}", exc_info=True)
        raise


# ==================== CrewAI Agent ====================

invoice_agent = Agent(
    role='Comprehensive Invoice Processing Specialist',
    goal="""Process invoices end-to-end: extract text, structure data, fetch database records,
    perform intelligent matching, generate reconciliation reports, and store results.""",
    backstory="""You are an expert invoice processing specialist with comprehensive skills in:

    1. OCR & Text Extraction: You can extract text from scanned PDFs with high accuracy
    2. Data Structuring: You intelligently parse messy text into structured invoice data
    3. Database Operations: You efficiently fetch and manage PO and GRN data
    4. Data Filtering: You filter data by release numbers and then by semantic similarity
    5. Reconciliation: You match invoice items to PO/GRN data and identify discrepancies
    6. Data Persistence: You store reports in databases for future reference

    You work methodically through each step, ensuring data quality and accuracy at every stage.
    You leverage AI and semantic analysis to handle variations in item descriptions.
    You provide clear, actionable insights in your reconciliation reports.""",

    tools=[
        extract_text_with_ocr,
        structure_info_with_llm,
        fetch_po_grn_details,
        filter_po_data_on_release,
        filter_by_desc_similarity,
        generate_report_with_llm,
        store_report_in_db
    ],
    verbose=True,
    allow_delegation=False
)

logger.info("Invoice processing agent created successfully")

# ==================== Single Sequential Task ====================

invoice_processing_task = Task(
    description="""Process the invoice PDF through the complete workflow:

    **Step 1: OCR Extraction**
    - Extract all text from the PDF located at {pdf_path}
    - Use the 'Extract Text with OCR' tool
    - Capture text from all pages

    **Step 2: Data Structuring**
    - Take the OCR text and structure it using 'Structure Invoice Information' tool
    - Extract invoice number, date, PO number, release number
    - Extract vendor details and all line items
    - Extract tax details and financial totals

    **Step 3: Database Fetching**
    - Extract PO number from structured data
    - Use 'Fetch PO and GRN Details' tool to get database records

    **Step 4: Release Filtering**
    - Filter PO data by release number from invoice
    - Use 'Filter PO Data on Release' tool

    **Step 5: Semantic Filtering**
    - Extract invoice item descriptions from structured data
    - Use 'Filter by Description Similarity' tool to match with PO items
    - This reduces the PO dataset to most relevant items

    **Step 6: Reconciliation**
    - Combine invoice, filtered PO, and GRN data
    - Use 'Generate Reconciliation Report' tool
    - Create detailed mapping of invoice items to PO items
    - Identify matched and unmatched items
    - Generate summary statistics

    **Step 7: Storage**
    - Store the complete report using 'Store Report in Database' tool
    - Save both summary and line-item details

    Execute each step in sequence, passing data from one step to the next.
    Handle errors gracefully and log progress at each stage.""",

    expected_output="""A comprehensive processing result including:

    1. Extracted OCR text from the invoice
    2. Structured invoice data with all fields
    3. Fetched PO and GRN data from database
    4. Filtered PO data matching release number and item descriptions
    5. Complete reconciliation report with:
       - Invoice number, PO number, GRN number, vendor name
       - Line-by-line item mapping (MAPPED/UNMAPPED status)
       - Remarks explaining match quality
       - Summary with total items, mapped count, unmapped count
    6. Confirmation of database storage

    All intermediate results saved to output files with timestamps.""",

    agent=invoice_agent,
    output_file=f"outputs/complete_invoice_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
)

logger.info("Invoice processing task created successfully")

# ==================== Main Crew ====================

logger.info("Creating invoice processing crew...")

invoice_crew = Crew(
    agents=[invoice_agent],
    tasks=[invoice_processing_task],
    process=Process.sequential,
    verbose=True
)

logger.info("Crew created successfully")

# ==================== Main Execution ====================

if __name__ == "__main__":
    try:
        logger.info("Starting Invoice Processing Pipeline with CrewAI")

        # Create outputs directory
        os.makedirs("outputs", exist_ok=True)
        os.makedirs("reports", exist_ok=True)

        pdf_path = "invoice_data/kobian.pdf"

        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            sys.exit(1)

        logger.info(f"Processing invoice: {pdf_path}")

        logger.info("Starting execution...")
        result = invoice_crew.kickoff(inputs={"pdf_path": pdf_path})


        logger.info("Invoice Processing Completed Successfully!")
        logger.info(f"Final Result:\n{result}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)