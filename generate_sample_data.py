"""
Sample Data Generator for E-Commerce Business Analytics Dashboard
Generates a realistic e-commerce transaction dataset modeled on the UCI Online Retail dataset.
Contains ~12,000 transactions over 14 months with realistic seasonality, pricing, repeat customers,
and geographic distributions.
"""

import os
import random
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def generate_ecommerce_dataset(output_path="data/online_retail_sample.csv", num_records=12000, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    products = [
        {"code": "85123A", "desc": "WHITE HANGING HEART T-LIGHT HOLDER", "cat": "Home & Living", "price": 2.55, "weight": 0.08},
        {"code": "71053", "desc": "WHITE METAL LANTERN", "cat": "Home & Living", "price": 3.39, "weight": 0.04},
        {"code": "84406B", "desc": "CREAM CUPID HEARTS COAT HANGER", "cat": "Home & Living", "price": 2.75, "weight": 0.03},
        {"code": "84029G", "desc": "KNITTED UNION FLAG HOT WATER BOTTLE", "cat": "Accessories & Gifts", "price": 3.39, "weight": 0.03},
        {"code": "84029E", "desc": "RED WOOLLY HOTTIE WHITE HEART", "cat": "Accessories & Gifts", "price": 3.39, "weight": 0.03},
        {"code": "22752", "desc": "SET 7 BABUSHKA NESTING BOXES", "cat": "Home & Living", "price": 7.65, "weight": 0.03},
        {"code": "21730", "desc": "GLASS STAR FROSTED T-LIGHT HOLDER", "cat": "Home & Living", "price": 4.25, "weight": 0.02},
        {"code": "22633", "desc": "HAND WARMER UNION JACK", "cat": "Accessories & Gifts", "price": 1.85, "weight": 0.04},
        {"code": "22632", "desc": "HAND WARMER RED POLKADOT", "cat": "Accessories & Gifts", "price": 1.85, "weight": 0.04},
        {"code": "84879", "desc": "ASSORTED COLOUR BIRD ORNAMENT", "cat": "Home & Living", "price": 1.69, "weight": 0.07},
        {"code": "22745", "desc": "POPPY'S PLAYHOUSE BEDROOM", "cat": "Toys & Kids", "price": 2.10, "weight": 0.02},
        {"code": "22748", "desc": "POPPY'S PLAYHOUSE KITCHEN", "cat": "Toys & Kids", "price": 2.10, "weight": 0.02},
        {"code": "22423", "desc": "REGENCY CAKESTAND 3 TIER", "cat": "Kitchen & Dining", "price": 12.75, "weight": 0.09},
        {"code": "22720", "desc": "SET OF 3 CAKE TINS PANTRY DESIGN", "cat": "Kitchen & Dining", "price": 4.95, "weight": 0.06},
        {"code": "22197", "desc": "POPCORN HOLDER", "cat": "Kitchen & Dining", "price": 0.85, "weight": 0.05},
        {"code": "21212", "desc": "PACK OF 72 RETROSPOT CAKE CASES", "cat": "Kitchen & Dining", "price": 0.55, "weight": 0.05},
        {"code": "22383", "desc": "LUNCH BAG SUKI DESIGN", "cat": "Kitchen & Dining", "price": 1.65, "weight": 0.04},
        {"code": "20725", "desc": "LUNCH BAG RED RETROSPOT", "cat": "Kitchen & Dining", "price": 1.65, "weight": 0.05},
        {"code": "20727", "desc": "LUNCH BAG BLACK SKULL.", "cat": "Kitchen & Dining", "price": 1.65, "weight": 0.03},
        {"code": "20728", "desc": "LUNCH BAG CARS BLUE", "cat": "Kitchen & Dining", "price": 1.65, "weight": 0.03},
        {"code": "22382", "desc": "LUNCH BAG SPACEBOY DESIGN", "cat": "Kitchen & Dining", "price": 1.65, "weight": 0.03},
        {"code": "22960", "desc": "JAM MAKING SET WITH JARS", "cat": "Kitchen & Dining", "price": 4.25, "weight": 0.03},
        {"code": "23084", "desc": "RABBIT NIGHT LIGHT", "cat": "Home & Living", "price": 2.08, "weight": 0.05},
        {"code": "23203", "desc": "JUMBO BAG DOILEY PATTERNS", "cat": "Storage & Organization", "price": 2.08, "weight": 0.03},
        {"code": "85099B", "desc": "JUMBO BAG RED RETROSPOT", "cat": "Storage & Organization", "price": 2.08, "weight": 0.08},
        {"code": "23199", "desc": "JUMBO BAG APPLES", "cat": "Storage & Organization", "price": 2.08, "weight": 0.03},
        {"code": "22386", "desc": "JUMBO BAG PINK POLKADOT", "cat": "Storage & Organization", "price": 2.08, "weight": 0.03},
        {"code": "22411", "desc": "JUMBO SHOPPER VINTAGE RED PAISLEY", "cat": "Storage & Organization", "price": 2.08, "weight": 0.02},
        {"code": "47566", "desc": "PARTY BUNTING", "cat": "Accessories & Gifts", "price": 4.95, "weight": 0.05},
        {"code": "22178", "desc": "VICTORIAN GLASS HANGING T-LIGHT", "cat": "Home & Living", "price": 1.25, "weight": 0.02},
        {"code": "22356", "desc": "CHARLOTTE BAG PINK POLKADOT", "cat": "Storage & Organization", "price": 0.85, "weight": 0.02},
        {"code": "21931", "desc": "JUMBO STORAGE BAG SUKI", "cat": "Storage & Organization", "price": 2.08, "weight": 0.02},
        {"code": "22629", "desc": "SPACEBOY LUNCH BOX", "cat": "Kitchen & Dining", "price": 1.95, "weight": 0.02},
        {"code": "22630", "desc": "DOLLY GIRL LUNCH BOX", "cat": "Kitchen & Dining", "price": 1.95, "weight": 0.02},
        {"code": "POST", "desc": "POSTAGE SERVICE UK & INT", "cat": "Service", "price": 18.00, "weight": 0.01},
        {"code": "M", "desc": "MANUAL TRANSACTION OVERRIDE", "cat": "Service", "price": 35.00, "weight": 0.005},
    ]

    countries = [
        ("United Kingdom", 0.82),
        ("Germany", 0.05),
        ("France", 0.04),
        ("EIRE", 0.02),
        ("Spain", 0.015),
        ("Netherlands", 0.015),
        ("Belgium", 0.01),
        ("Switzerland", 0.01),
        ("Australia", 0.01),
        ("USA", 0.005),
        ("Singapore", 0.005),
    ]

    country_names = [c[0] for c in countries]
    country_weights = [c[1] for c in countries]

    # Customers: 650 active customers with varying frequency profiles (Whales, Regulars, One-off)
    customers = [12000 + i for i in range(650)]
    customer_weights = np.ones(len(customers))
    customer_weights[:30] = 6.0   # Whales
    customer_weights[30:120] = 3.0 # Regulars
    customer_weights /= customer_weights.sum()

    start_date = datetime(2023, 1, 1, 8, 30)
    end_date = datetime(2024, 2, 28, 18, 0)
    total_days = (end_date - start_date).days

    # Pre-build daily probability weights to model realistic retail seasonality (Q4 holiday surge)
    day_weights = []
    for d in range(total_days + 1):
        cur_date = start_date + timedelta(days=d)
        w = 1.0
        # Holiday season surge: Nov & Dec
        if cur_date.month in [11, 12]:
            w = 2.4
        elif cur_date.month in [9, 10]:
            w = 1.4
        elif cur_date.month in [1, 2]:
            w = 0.85
        # Weekday vs Weekend pattern: retail orders peak Tue-Thu
        if cur_date.weekday() in [1, 2, 3]:
            w *= 1.25
        elif cur_date.weekday() == 6: # Sunday
            w *= 0.7
        day_weights.append(w)
    
    day_weights = np.array(day_weights)
    day_weights /= day_weights.sum()

    records = []
    invoice_seq = 536365
    num_invoices = num_records // 3  # ~4000 invoices averaging 3 items each

    for _ in range(num_invoices):
        invoice_seq += 1
        
        # Sample date using seasonal distribution
        day_offset = int(np.random.choice(len(day_weights), p=day_weights))
        hour = random.randint(8, 19)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        inv_date = start_date + timedelta(days=day_offset, hours=hour, minutes=minute, seconds=second)

        # Assign customer (14% guest checkout)
        is_guest = random.random() < 0.14
        customer_id = None if is_guest else int(np.random.choice(customers, p=customer_weights))
        
        country = random.choices(country_names, weights=country_weights)[0]

        # Is this a cancellation? (~2.5% of invoices)
        is_cancellation = random.random() < 0.025
        invoice_str = f"C{invoice_seq}" if is_cancellation else str(invoice_seq)

        # 1 to 6 items per invoice
        num_items = random.choices([1, 2, 3, 4, 5, 8], weights=[0.4, 0.25, 0.15, 0.1, 0.06, 0.04])[0]
        prod_weights = [p["weight"] for p in products]
        selected_prods = random.choices(products, weights=prod_weights, k=num_items)

        for prod in selected_prods:
            qty_base = random.choices([1, 2, 3, 4, 6, 12, 24, 48], weights=[0.35, 0.25, 0.12, 0.1, 0.08, 0.05, 0.03, 0.02])[0]
            quantity = -qty_base if is_cancellation else qty_base
            
            # Slight price fluctuation / wholesale discount
            unit_price = round(prod["price"] * (0.95 if qty_base >= 12 else 1.0), 2)

            records.append({
                "InvoiceNo": invoice_str,
                "StockCode": prod["code"],
                "Description": prod["desc"],
                "Quantity": quantity,
                "InvoiceDate": inv_date.strftime("%Y-%m-%d %H:%M:%S"),
                "UnitPrice": unit_price,
                "CustomerID": "" if customer_id is None else str(customer_id),
                "Country": country,
                "Category": prod["cat"]
            })

    df = pd.DataFrame(records)

    # Inject minor realistic anomalies: 0.5% duplicate rows
    dups = df.sample(frac=0.005, random_state=random_seed)
    df = pd.concat([df, dups], ignore_index=True)

    # Sort chronologically
    df["InvoiceDate_dt"] = pd.to_datetime(df["InvoiceDate"])
    df = df.sort_values("InvoiceDate_dt").reset_index(drop=True)
    df = df.drop(columns=["InvoiceDate_dt"])

    df.to_csv(output_path, index=False)
    print(f"SUCCESS: Generated sample dataset at {output_path} with {len(df):,} rows.")
    return output_path

if __name__ == "__main__":
    generate_ecommerce_dataset()
