import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from crewai import Agent, Task, Crew, Process
from config.logger import setup_logging
from agent_tools.ocr_text_extraction import extract_text_with_ocr,structure_info_with_llm
from agent_tools.fetch_and_filter_data import fetch_po_grn_details,filter_po_data_on_release,filter_by_desc_similarity
from agent_tools.generate_and_store_report import generate_report_with_llm,store_report_in_db

logger = setup_logging()

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
        db_path = "reports/invoice_reports.db"

        if not os.path.exists(pdf_path):
            logger.error(f"PDF file not found: {pdf_path}")
            sys.exit(1)

        logger.info(f"Processing invoice: {pdf_path}")

        logger.info("Starting execution...")
        result = invoice_crew.kickoff(inputs={"pdf_path": pdf_path,
                                              "db_path": db_path})


        logger.info("Invoice Processing Completed Successfully!")
        logger.info(f"Final Result:\n{result}")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)