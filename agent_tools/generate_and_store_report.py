import os
import json
import sqlite3

from crewai.tools import tool
from config.logger import setup_logging

from langchain_openai import ChatOpenAI
from config.settings import settings

from models.mapping import MappingReport
api_key = settings.OPENAI_API_KEY
logger = setup_logging()

# ==================== Main Functions with @tool Decorator ====================
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
