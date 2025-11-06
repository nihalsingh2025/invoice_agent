import os
import cv2
import numpy as np
import tempfile

from crewai.tools import tool
from config.logger import setup_logging
from pdf2image import convert_from_path

from doctr.io import DocumentFile
from doctr.models import ocr_predictor

from langchain_openai import ChatOpenAI
from models.inovice import InvoiceSchema

from config.settings import settings

logger = setup_logging()

api_key = settings.OPENAI_API_KEY
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
