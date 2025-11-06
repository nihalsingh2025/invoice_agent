import os
from dotenv import load_dotenv
load_dotenv()

class Settings:

    ORACLE_DSN = os.getenv('ORACLE_DSN')
    ORACLE_USERNAME = os.getenv('ORACLE_USERNAME')
    ORACLE_PASSWORD = os.getenv('ORACLE_PASSWORD')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

settings = Settings()