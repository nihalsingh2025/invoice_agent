import logging
import os

def setup_logging():
    os.makedirs('logs', exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/invoice_processing.log'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)