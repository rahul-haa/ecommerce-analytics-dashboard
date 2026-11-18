"""
Machine Learning Module for E-Commerce Business Analytics Dashboard
Performs Customer Segmentation using K-Means clustering on RFM features,
evaluates cluster quality, generates 2D/3D visualizations, and translates clusters
into actionable business personas.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Tuple


def perform_kmeans_clustering(rfm_df: pd.DataFrame, n_clusters: int = 4, random_state: int = 42) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Applies K-Means clustering on Log-transformed and Standardized RFM features.
    """
    if len(rfm_df) < n_clusters:
        return rfm_df, {"error": "Not enough customer records for clustering."}

    # Features for clustering
    features = ["Recency", "Frequency", "Monetary"]
    X_raw = rfm_df[features].copy()

    # Log1p transformation to normalize skewed transaction distributions
    X_log = np.log1p(X_raw)

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    # K-Means model
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    # Compute silhouette score
    sil_score = float(silhouette_score(X_scaled, clusters)) if len(rfm_df) > n_clusters else 0.0

    # 2D PCA for clean dimensional projection
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)
    pca_var_ratio = pca.explained_variance_ratio_

    df_clustered = rfm_df.copy()
    df_clustered["Cluster"] = clusters
    df_clustered["PCA1"] = pca_coords[:, 0]
    df_clustered["PCA2"] = pca_coords[:, 1]

    # Compute cluster profile medians to assign meaningful business personas
    cluster_profiles = df_clustered.groupby("Cluster").agg(
        Count=("CustomerID_Clean", "count"),
        Recency_Median=("Recency", "median"),
        Frequency_Median=("Frequency", "median"),
        Monetary_Median=("Monetary", "median"),
        Total_Revenue=("Monetary", "sum")
    ).reset_index()

    # Derive persona names based on ranking relative to peers
    recency_ranks = cluster_profiles["Recency_Median"].rank(ascending=True)   # Low recency = recent = best
    freq_ranks = cluster_profiles["Frequency_Median"].rank(ascending=False)     # High freq = best
    monetary_ranks = cluster_profiles["Monetary_Median"].rank(ascending=False) # High money = best

    overall_score = recency_ranks + freq_ranks + monetary_ranks

    persona_map = {}
    descriptions = {}
    recommendations = {}

    # Sort clusters by overall rank
    sorted_clusters = cluster_profiles.sort_values(by="Total_Revenue", ascending=False)["Cluster"].tolist()
    
    # Pre-defined strategic persona templates
    persona_templates = [
        {
            "name": "High-Value VIPs & Wholesalers",
            "desc": "Top spenders with high repeat order frequency. Core revenue drivers.",
            "rec": "Provide dedicated account management, exclusive early product access, and tier-based loyalty rewards."
        },
        {
            "name": "Loyal Regular Shoppers",
            "desc": "Frequent buyers with steady transaction sizes and consistent engagement.",
            "rec": "Cross-sell complementary categories, recommend subscription bundles, and send seasonal catalogs."
        },
        {
            "name": "New / Recent Shoppers",
            "desc": "Recently acquired customers with low initial order count. High expansion potential.",
            "rec": "Onboard with welcome discount sequences, post-purchase follow-ups, and customer satisfaction surveys."
        },
        {
            "name": "At-Risk / Lapsed Customers",
            "desc": "Long time since last purchase with declining or stagnant engagement.",
            "rec": "Trigger automated win-back emails, limited-time discount incentives, or personalized 'We Miss You' surveys."
        },
        {
            "name": "Occasional Bargain Hunters",
            "desc": "Price-sensitive buyers who only purchase during sales or low-ticket promotions.",
            "rec": "Target with promotional flash sales, clearance notifications, and bundle minimum-order incentives."
        }
    ]

    for idx, c_id in enumerate(sorted_clusters):
        template = persona_templates[idx % len(persona_templates)]
        persona_map[c_id] = template["name"]
        descriptions[c_id] = template["desc"]
        recommendations[c_id] = template["rec"]

    df_clustered["Persona"] = df_clustered["Cluster"].map(persona_map)
    df_clustered["Cluster_Label"] = "Cluster " + df_clustered["Cluster"].astype(str) + ": " + df_clustered["Persona"]

    # Cluster summary
    summary = df_clustered.groupby(["Cluster", "Persona"]).agg(
        Customers=("CustomerID_Clean", "count"),
        Avg_Recency_Days=("Recency", "mean"),
        Avg_Frequency_Orders=("Frequency", "mean"),
        Avg_Monetary_Spend=("Monetary", "mean"),
        Total_Revenue=("Monetary", "sum")
    ).reset_index()

    total_rev = summary["Total_Revenue"].sum()
    summary["Revenue_Share_Pct"] = (summary["Total_Revenue"] / max(total_rev, 1)) * 100
    summary["Customer_Share_Pct"] = (summary["Customers"] / max(len(df_clustered), 1)) * 100
    summary["Description"] = summary["Cluster"].map(descriptions)
    summary["Actionable_Strategy"] = summary["Cluster"].map(recommendations)

    ml_metrics = {
        "silhouette_score": round(sil_score, 3),
        "inertia": float(kmeans.inertia_),
        "pca_variance": [round(float(v) * 100, 1) for v in pca_var_ratio],
        "cluster_summary": summary,
        "n_clusters": n_clusters,
    }

    return df_clustered, ml_metrics


def build_cluster_scatter_3d(df_clustered: pd.DataFrame, theme: str = "dark") -> go.Figure:
    """
    Creates a responsive 3D Plotly scatter plot of RFM clusters.
    Uses pure Python lists to prevent binary buffer (bdata) serialization
    and ensures full WebGL rendering compatibility across all browsers.
    """
    fig = go.Figure()
    palette = ["#635bff", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#38bdf8"]
    is_dark = (theme == "dark")

    # Styling colors
    text_color = "#94a3b8" if is_dark else "#475569"
    grid_color = "#2d3248" if is_dark else "#e2e8f0"
    plane_bg = "#151828" if is_dark else "#f8fafc"
    marker_line = "#ffffff" if is_dark else "#0f172a"

    unique_personas = df_clustered["Persona"].unique()
    for i, persona in enumerate(unique_personas):
        sub = df_clustered[df_clustered["Persona"] == persona]
        color = palette[i % len(palette)]

        hover_texts = [
            f"<b>Customer:</b> {row['CustomerID_Clean']}<br>"
            f"<b>Segment:</b> {persona}<br>"
            f"<b>Recency:</b> {row['Recency']:.0f} days<br>"
            f"<b>Frequency:</b> {row['Frequency']:.0f} orders<br>"
            f"<b>Monetary:</b> ${row['Monetary']:,.2f}"
            for _, row in sub.iterrows()
        ]

        fig.add_trace(go.Scatter3d(
            x=sub["Recency"].tolist(),
            y=sub["Frequency"].tolist(),
            z=sub["Monetary"].tolist(),
            mode="markers",
            name=persona,
            text=hover_texts,
            hoverinfo="text",
            marker=dict(
                size=5,
                color=color,
                opacity=0.85,
                line=dict(width=0.6, color=marker_line),
            ),
        ))

    fig.update_layout(
        height=560,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=text_color, size=11),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=text_color),
        ),
        scene=dict(
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.65, y=1.65, z=1.35),
            ),
            xaxis=dict(
                title="Recency (Days)",
                gridcolor=grid_color,
                zerolinecolor=grid_color,
                showbackground=True,
                backgroundcolor=plane_bg,
                color=text_color,
            ),
            yaxis=dict(
                title="Order Frequency",
                gridcolor=grid_color,
                zerolinecolor=grid_color,
                showbackground=True,
                backgroundcolor=plane_bg,
                color=text_color,
            ),
            zaxis=dict(
                title="Monetary ($)",
                gridcolor=grid_color,
                zerolinecolor=grid_color,
                showbackground=True,
                backgroundcolor=plane_bg,
                color=text_color,
            ),
        ),
    )
    return fig


def build_pca_2d_scatter(df_clustered: pd.DataFrame, pca_variance: list, theme: str = "dark") -> go.Figure:
    """
    Creates a 2D PCA cluster projection plot using native lists to prevent bdata serialization.
    """
    fig = go.Figure()
    palette = ["#635bff", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#38bdf8"]
    is_dark = (theme == "dark")

    text_color = "#94a3b8" if is_dark else "#475569"
    grid_color = "#232636" if is_dark else "#e2e8f0"
    marker_line = "#ffffff" if is_dark else "#0f172a"

    unique_personas = df_clustered["Persona"].unique()
    for i, persona in enumerate(unique_personas):
        sub = df_clustered[df_clustered["Persona"] == persona]
        color = palette[i % len(palette)]

        hover_texts = [
            f"<b>Customer:</b> {row['CustomerID_Clean']}<br>"
            f"<b>Segment:</b> {persona}<br>"
            f"<b>PCA 1:</b> {row['PCA1']:.2f}<br>"
            f"<b>PCA 2:</b> {row['PCA2']:.2f}<br>"
            f"<b>Recency:</b> {row['Recency']:.0f} days<br>"
            f"<b>Frequency:</b> {row['Frequency']:.0f} orders<br>"
            f"<b>Monetary:</b> ${row['Monetary']:,.2f}"
            for _, row in sub.iterrows()
        ]

        fig.add_trace(go.Scatter(
            x=sub["PCA1"].tolist(),
            y=sub["PCA2"].tolist(),
            mode="markers",
            name=persona,
            text=hover_texts,
            hoverinfo="text",
            marker=dict(
                size=7,
                color=color,
                opacity=0.85,
                line=dict(width=0.6, color=marker_line),
            ),
        ))

    fig.update_layout(
        height=480,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            title=f"PCA Component 1 ({pca_variance[0]}% variance)",
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            automargin=True,
            color=text_color,
        ),
        yaxis=dict(
            title=f"PCA Component 2 ({pca_variance[1]}% variance)",
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            automargin=True,
            color=text_color,
        ),
        font=dict(family="Inter, sans-serif", color=text_color),
        margin=dict(l=60, r=20, t=30, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=text_color),
        ),
    )
    return fig

