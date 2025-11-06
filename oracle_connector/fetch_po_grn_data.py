import pandas as pd
import json
from datetime import datetime
from utils.logger import setup_logger
from oracle_connector.fetch_oracle_base import DataFetcherOracleBase

logger = setup_logger("invoice_processing")

class DataFetcherOraclePOGRN(DataFetcherOracleBase):
    """
    Fetches both Purchase Order (PO) and Goods Receipt Note (GRN) data from Oracle.
    """

    def __init__(self, psswrd, user_name, dsn):
        super().__init__(psswrd, user_name, dsn)

    # ------------------ FETCH PO DATA ------------------ #
    def fetch_po_data(self, po_number: int) -> str :
        """Fetch Purchase Order (PO) details for a given PO number."""
        try:
            if not self.connection:
                raise ConnectionError("Database connection is not established.")

            logger.info(f"Fetching PO data for PO Number: {po_number}")

            # Set Oracle context
            self.cursor.execute("""
                                BEGIN
                                    mo_global.set_policy_context('S','82');
                                END;
                                """)
            self.connection.commit()

            query = """
                    SELECT ph.segment1              AS po_number, \
                           ph.po_header_id, \
                           por.RELEASE_NUM, \
                           ph.type_lookup_code      AS po_type, \
                           ph.authorization_status, \
                           ph.creation_date         AS po_date, \
                           pv.vendor_name, \
                           pv.segment1              AS vendor_code, \
                           pvs.vendor_site_code, \
                           pl.line_num, \
                           pl.po_line_id, \
                           msi.segment1             AS item_code, \
                           msi.description          AS item_description, \
                           pl.item_id, \
                           pl.category_id, \
                           pl.unit_price, \
                           pl.quantity, \
                           pl.unit_meas_lookup_code AS uom, \
                           pll.ship_to_location_id, \
                           pll.need_by_date, \
                           pll.quantity             AS line_location_qty, \
                           pd.PO_DISTRIBUTION_ID, \
                           pd.code_combination_id, \
                           pd.AMOUNT_BILLED, \
                           pd.ENCUMBERED_AMOUNT, \
                           pd.RECOVERABLE_TAX
                    FROM po_headers_all ph, \
                         po_lines_all pl, \
                         po_line_locations_all pll, \
                         po_distributions_all pd, \
                         po_vendors pv, \
                         po_vendor_sites_all pvs, \
                         mtl_system_items_b msi, \
                         po_releases_all por
                    WHERE ph.po_header_id = pl.po_header_id
                      AND pl.po_line_id = pll.po_line_id
                      AND ph.PO_HEADER_ID = por.PO_HEADER_ID(+)
                      AND pll.line_location_id = pd.line_location_id
                      AND ph.vendor_id = pv.vendor_id
                      AND ph.vendor_site_id = pvs.vendor_site_id
                      AND msi.inventory_item_id(+) = pl.item_id
                      AND NVL(msi.organization_id, '83') = '83'
                      and ph.PO_HEADER_ID = por.PO_HEADER_ID(+)
                      AND ph.segment1 = :po_number \
                    """

            # Execute query safely
            self.cursor.execute(query, {"po_number": po_number})
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]

            data = pd.DataFrame(rows, columns=columns)

            if data.empty:
                logger.warning(f"No PO records found for {po_number}")
                return "No Data Available"

            data = data.drop(columns=[
                            "AUTHORIZATION_STATUS",
                            "PO_LINE_ID",
                            "ITEM_ID",
                            "CATEGORY_ID",
                            "SHIP_TO_LOCATION_ID",
                            "PO_DISTRIBUTION_ID",
                            "CODE_COMBINATION_ID",
                        ])

            data = data.drop_duplicates()

            result_str = data.to_json(orient="records", indent=2)
            logger.info(f"PO data successfully fetched and aggregated for {po_number}")
            return result_str

        except Exception as e:
            raise RuntimeError(f" Failed to fetch PO data for {po_number}: {e}")

    # ------------------ FETCH GRN DATA ------------------ #
    def fetch_grn_data(self, po_number: int) -> str:
        """Fetch Goods Receipt Note (GRN) details for a given PO number."""
        try:
            if not self.connection:
                raise ConnectionError("Database connection is not established.")

            logger.info(f"Fetching GRN data for PO Number: {po_number}")

            self.cursor.execute("""
                                BEGIN
                                    mo_global.set_policy_context('S','82');
                                END;
                                """)
            self.connection.commit()

            query = """
                    SELECT a.po_header_id,
                           a.org_id,
                           a.segment1                                                   AS po_no,
                           a.creation_date                                              AS order_date,
                           b.segment1                                                   AS vendor_code,
                           b.vendor_name,
                           c.vendor_site_code,
                           c.address_line1,
                           c.address_line2,
                           c.address_line3,
                           c.city || ' ' || c.state || ' ' || c.zip || ' ' || c.country AS city_country,
                           b.vendor_id,
                           c.vendor_site_id,
                           d.receipt_num                                                AS grn_no,
                           d.creation_date                                              AS receipt_date,
                           d.shipment_num || ' ' || d.shipped_date                      AS supplier_inv_date,
                           d.num_of_containers                                          AS container_id,
                           d.comments                                                   AS remarks,
                           d.shipment_header_id,
                           f.shipment_line_id,
                           g.segment1                                                   AS item_code,
                           f.item_id,
                           g.description                                                AS item_description,
                           g.primary_uom_code                                           AS uom,
                           e.quantity,
                           d.receipt_num,
                           h.subinventory,
                           f.quantity_received,
                           f.line_num                                                   AS receipt_line_num
                    FROM po_headers_all a,
                         po_vendors b,
                         po_vendor_sites_all c,
                         rcv_shipment_headers_v d,
                         po_lines_all e,
                         rcv_shipment_lines f,
                         mtl_system_items g,
                         rcv_transactions h
                    WHERE a.vendor_id = b.vendor_id
                      AND a.vendor_site_id = c.vendor_site_id
                      AND b.vendor_id = c.vendor_id
                      AND a.vendor_id = d.vendor_id
                      AND a.vendor_site_id = d.vendor_site_id
                      AND a.po_header_id = e.po_header_id
                      AND d.shipment_header_id = f.shipment_header_id
                      AND f.po_header_id = e.po_header_id
                      AND f.po_line_id = e.po_line_id
                      AND f.item_id IS NOT NULL
                      AND f.po_header_id = a.po_header_id
                      AND f.item_id = g.inventory_item_id(+)
                      AND f.to_organization_id = g.organization_id(+)
                      AND f.shipment_header_id = h.shipment_header_id
                      AND f.shipment_line_id = h.shipment_line_id
                      AND UPPER(h.transaction_type) = 'DELIVER'
                      AND h.inspection_status_code != 'REJECTED'
              AND TO_NUMBER(a.segment1) = NVL(TO_NUMBER(:p_lpo_numbers), TO_NUMBER(a.segment1)) 
                    """

            self.cursor.execute(query, {"p_lpo_numbers": po_number})
            rows = self.cursor.fetchall()
            columns = [desc[0] for desc in self.cursor.description]

            data = pd.DataFrame(rows, columns=columns)
            if data.empty:
                logger.warning(f"No GRN records found for PO {po_number}")
                return "No Data Available"

            data = data.drop(columns=[
                "ORG_ID",
                "CONTAINER_ID",
                "SHIPMENT_HEADER_ID",
                "SHIPMENT_LINE_ID",
                "ITEM_ID",
            ])

            data = data.drop_duplicates()
            result_str = data.to_json(orient="records", indent=2)

            logger.info(f"GRN data successfully fetched and aggregated for {po_number}")
            return result_str

        except Exception as e:
            raise RuntimeError(f" Failed to fetch GRN data for {po_number}: {e}")
