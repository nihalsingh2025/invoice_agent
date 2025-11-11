import json
import pandas as pd

from crewai.tools import tool
from config.logger import setup_logging

from oracle_connector.fetch_po_grn_data import DataFetcherOraclePOGRN
from config.settings import settings

from openai import OpenAI, APIError
from sklearn.metrics.pairwise import cosine_similarity

dsn = settings.ORACLE_DSN
username = settings.ORACLE_USERNAME
password = settings.ORACLE_PASSWORD
api_key = settings.OPENAI_API_KEY

logger = setup_logging()

# ==================== Main Functions with @tool Decorator ====================

@tool("Fetch PO and GRN Details")
def fetch_po_grn_details(po_number: int) -> str:
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
def filter_po_data_on_release(data_dict: dict, release_num: str) -> str:
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
            return po_df.to_json(orient="records")
        else:
            po_df = po_df[po_df["RELEASE_NUM"].astype(str) == str(release_num)]
            logger.info(f"After filtering: {po_df.shape[0]} rows remaining")
            return po_df.to_json(orient="records")

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
        df_json: DataFrame to filter (PO data)
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

