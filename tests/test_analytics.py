"""
Unit and Integration Tests for E-Commerce Business Analytics Engine
"""

import unittest
import os
import sys

# Ensure root directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import numpy as np

from src.data_loader import load_dataset, validate_dataset_schema, clean_and_process_data
from src.analytics import (
    compute_overview_kpis,
    calculate_period_delta,
    get_sales_analytics,
    get_product_analytics,
    compute_rfm_segmentation,
    get_customer_analytics,
    generate_sparkline_fig
)
from src.ml_segmentation import perform_kmeans_clustering, build_cluster_scatter_3d, build_pca_2d_scatter
from src.insights_engine import generate_business_insights, generate_business_recommendations

class TestEcommerceAnalytics(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sample_path = "data/online_retail_sample.csv"
        assert os.path.exists(cls.sample_path), "Sample dataset must exist."
        cls.clean_df, cls.cancelled_df, cls.health, cls.err = load_dataset(cls.sample_path)
        assert cls.err is None, f"Dataset load failed: {cls.err}"

    def test_data_loader_and_schema(self):
        self.assertGreater(len(self.clean_df), 1000)
        self.assertGreater(len(self.cancelled_df), 10)
        self.assertIn("Revenue", self.clean_df.columns)
        self.assertIn("InvoiceDate", self.clean_df.columns)
        self.assertIn("CustomerID_Clean", self.clean_df.columns)
        self.assertTrue((self.clean_df["Quantity"] > 0).all())
        self.assertTrue((self.clean_df["UnitPrice"] > 0).all())
        self.assertFalse(self.clean_df["IsCancelled"].any())

    def test_kpi_calculations(self):
        kpis = compute_overview_kpis(self.clean_df)
        self.assertGreater(kpis["revenue"], 0)
        self.assertGreater(kpis["orders"], 0)
        self.assertGreater(kpis["customers"], 0)
        self.assertGreater(kpis["aov"], 0)
        self.assertNotEqual(kpis["best_seller"], "N/A")

    def test_period_delta(self):
        delta, badge, is_pos = calculate_period_delta(110, 100)
        self.assertEqual(delta, 10.0)
        self.assertEqual(badge, "+10.0%")
        self.assertTrue(is_pos)

        delta2, badge2, is_pos2 = calculate_period_delta(90, 100)
        self.assertEqual(delta2, -10.0)
        self.assertEqual(badge2, "-10.0%")
        self.assertFalse(is_pos2)

    def test_sales_analytics(self):
        sales = get_sales_analytics(self.clean_df)
        self.assertFalse(sales["monthly_sales"].empty)
        self.assertFalse(sales["daily_sales"].empty)
        self.assertFalse(sales["country_sales"].empty)
        self.assertFalse(sales["hourly_sales"].empty)
        self.assertFalse(sales["dow_sales"].empty)

    def test_product_analytics(self):
        prods = get_product_analytics(self.clean_df)
        self.assertEqual(len(prods["top_revenue"]), 10)
        self.assertEqual(len(prods["top_quantity"]), 10)
        self.assertEqual(len(prods["top_profit"]), 10)
        self.assertFalse(prods["category_sales"].empty)
        self.assertGreater(prods["pct_skus_for_80"], 0)

    def test_rfm_and_customer_analytics(self):
        cust = get_customer_analytics(self.clean_df)
        rfm_df = cust["rfm_df"]
        self.assertIn("R_Score", rfm_df.columns)
        self.assertIn("F_Score", rfm_df.columns)
        self.assertIn("M_Score", rfm_df.columns)
        self.assertIn("Segment", rfm_df.columns)
        self.assertFalse(cust["segment_summary"].empty)
        self.assertFalse(cust["top_customers"].empty)

    def test_ml_kmeans_clustering(self):
        rfm_df, _ = compute_rfm_segmentation(self.clean_df)
        clustered_df, ml_meta = perform_kmeans_clustering(rfm_df, n_clusters=4)
        self.assertEqual(ml_meta["n_clusters"], 4)
        self.assertIn("Cluster", clustered_df.columns)
        self.assertIn("Persona", clustered_df.columns)
        self.assertGreater(ml_meta["silhouette_score"], -1.0)
        self.assertEqual(len(ml_meta["cluster_summary"]), 4)

    def test_ml_visualizations(self):
        import plotly.io as pio
        rfm_df, _ = compute_rfm_segmentation(self.clean_df)
        clustered_df, ml_meta = perform_kmeans_clustering(rfm_df, n_clusters=4)
        fig_3d = build_cluster_scatter_3d(clustered_df, theme="dark")
        self.assertIsNotNone(fig_3d)
        self.assertEqual(len(fig_3d.data), 4)
        # Ensure no bdata serialization issues
        spec_3d = pio.to_json(fig_3d, validate=False)
        self.assertNotIn("bdata", spec_3d)

        fig_2d = build_pca_2d_scatter(clustered_df, ml_meta["pca_variance"], theme="dark")
        self.assertIsNotNone(fig_2d)
        self.assertEqual(len(fig_2d.data), 4)
        spec_2d = pio.to_json(fig_2d, validate=False)
        self.assertNotIn("bdata", spec_2d)

    def test_insights_and_recommendations(self):
        rfm_df, _ = compute_rfm_segmentation(self.clean_df)
        insights = generate_business_insights(self.clean_df, rfm_df)
        self.assertGreaterEqual(len(insights), 4)

        recs = generate_business_recommendations(self.clean_df, rfm_df)
        self.assertGreaterEqual(len(recs), 3)

    def test_sparkline_figure(self):
        fig = generate_sparkline_fig([10, 20, 15, 35, 40], is_positive=True)
        self.assertIsNotNone(fig)

if __name__ == "__main__":
    unittest.main()
