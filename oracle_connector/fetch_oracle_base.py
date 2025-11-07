import pandas as pd
import oracledb
from config.logger import setup_logging
from pathlib import Path

lib_path = r"E:\oracle\instantclient_23_9"
oracledb.init_oracle_client(lib_dir=str(lib_path))

logger=setup_logging()

class DataFetcherOracleBase:
    
    def __init__(self,psswrd,user_name,dsn):
        self.connection = None
        self.cursor = None
        self.psswrd=psswrd
        self.user_name=user_name
        self.dsn=dsn

    def connect(self):
        try:
            self.connection = oracledb.connect(
                user=self.user_name,
                password=self.psswrd,
                dsn=self.dsn
            )
            self.cursor = self.connection.cursor()
            logger.info("connection established")
        except Exception as e:
            logger.exception(f"Failed to connect to Oracle DB: {e}")
            raise ConnectionError(f"Failed to connect to Oracle DB: {e}")    

    def disconnect(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
                