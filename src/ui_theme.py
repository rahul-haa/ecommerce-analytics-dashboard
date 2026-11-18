"""
UI Theme and Custom Styling for E-Commerce Business Analytics Dashboard
Provides full support for both Dark Mode and Light Mode with high contrast and executive typography.
Eliminates amateur emojis/boxes and adopts clean, minimal enterprise glyphs (e.g. ↗, ⓘ, ›, •, ✦).
"""

def get_custom_css(theme="dark"):
    """
    Returns custom CSS tailored to replicate the Stripe Dashboard aesthetic
    in both Dark and Light modes without text clipping or contrast issues.
    """
    if theme == "dark":
        bg_color = "#0c0d12"
        surface_color = "#151722"
        surface_hover = "#1c1e2d"
        card_bg = "#181a26"
        card_border = "rgba(255, 255, 255, 0.08)"
        text_primary = "#f8fafc"
        text_secondary = "#94a3b8"
        text_muted = "#64748b"
        accent_color = "#635bff"  # Stripe purple
        accent_light = "#818cf8"
        banner_bg = "linear-gradient(135deg, rgba(99, 91, 255, 0.12) 0%, rgba(24, 26, 38, 0.8) 100%)"
        banner_border = "rgba(99, 91, 255, 0.3)"
        notice_bg = "rgba(24, 26, 38, 0.7)"
        cell_bg = "rgba(0, 0, 0, 0.25)"
        tag_green_bg = "rgba(34, 197, 94, 0.15)"
        tag_green_text = "#4ade80"
        tag_red_bg = "rgba(239, 68, 68, 0.15)"
        tag_red_text = "#f87171"
        table_border = "#232636"
        card_shadow = "0 4px 12px rgba(0, 0, 0, 0.25)"
        input_bg = "#181a26"
        input_border = "rgba(255, 255, 255, 0.12)"
    else:
        bg_color = "#f8fafc"
        surface_color = "#ffffff"
        surface_hover = "#f1f5f9"
        card_bg = "#ffffff"
        card_border = "#e2e8f0"
        text_primary = "#0f172a"
        text_secondary = "#334155"
        text_muted = "#64748b"
        accent_color = "#635bff"
        accent_light = "#4f46e5"
        banner_bg = "linear-gradient(135deg, rgba(99, 91, 255, 0.08) 0%, rgba(241, 245, 249, 0.9) 100%)"
        banner_border = "rgba(99, 91, 255, 0.25)"
        notice_bg = "#f1f5f9"
        cell_bg = "#f8fafc"
        tag_green_bg = "#dcfce7"
        tag_green_text = "#166534"
        tag_red_bg = "#fee2e2"
        tag_red_text = "#991b1b"
        table_border = "#e2e8f0"
        card_shadow = "0 2px 8px rgba(0, 0, 0, 0.06)"
        input_bg = "#ffffff"
        input_border = "#cbd5e1"

    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Resets */
    html, body, [class*="css"], .stApp {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: {bg_color} !important;
        color: {text_primary} !important;
    }}

    /* Container Spacing */
    .main .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 1400px;
    }}

    /* Clean Streamlit Headers */
    #MainMenu, footer, header {{
        visibility: hidden;
    }}

    /* Global Headings and Paragraphs */
    h1, h2, h3, h4, h5, h6 {{
        color: {text_primary} !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        letter-spacing: -0.02em;
    }}
    p, span, label {{
        color: {text_secondary};
    }}
    strong, b {{
        color: {text_primary} !important;
    }}

    /* Section Headings */
    .section-title {{
        font-size: 20px;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: {text_primary} !important;
        margin-top: 12px;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .section-subtitle {{
        font-size: 13px;
        color: {text_muted} !important;
        margin-top: 0px;
        margin-bottom: 18px;
    }}

    /* Professional Topbar */
    .stripe-topbar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 18px;
        background: {surface_color};
        border-radius: 12px;
        border: 1px solid {card_border};
        margin-bottom: 20px;
    }}
    .stripe-topbar-title {{
        font-size: 22px;
        font-weight: 700;
        color: {text_primary} !important;
        letter-spacing: -0.03em;
    }}

    /* Cards */
    .stripe-card {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 12px;
        padding: 18px 20px;
        position: relative;
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
        box-shadow: {card_shadow};
        height: 100%;
    }}
    .stripe-card:hover {{
        border-color: rgba(99, 91, 255, 0.4) !important;
    }}
    .stripe-card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }}
    .stripe-card-title {{
        font-size: 13px;
        font-weight: 500;
        color: {text_secondary} !important;
        display: flex;
        align-items: center;
        gap: 6px;
    }}
    .stripe-card-value {{
        font-size: 26px;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: {text_primary} !important;
        margin-bottom: 4px;
        line-height: 1.2;
    }}
    .stripe-card-subtext {{
        font-size: 12px;
        color: {text_muted} !important;
        display: flex;
        align-items: center;
        gap: 6px;
    }}

    /* Pill Badges */
    .stripe-pill-badge {{
        font-size: 11px;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-flex;
        align-items: center;
        gap: 4px;
        letter-spacing: 0.02em;
    }}
    .badge-positive {{
        background-color: {tag_green_bg} !important;
        color: {tag_green_text} !important;
    }}
    .badge-negative {{
        background-color: {tag_red_bg} !important;
        color: {tag_red_text} !important;
    }}

    /* Action Banner */
    .stripe-banner {{
        background: {banner_bg} !important;
        border: 1px solid {banner_border} !important;
        border-radius: 12px;
        padding: 18px 22px;
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }}
    .stripe-banner-tag {{
        font-size: 12px;
        font-weight: 600;
        color: {accent_light} !important;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}
    .stripe-banner-title {{
        font-size: 15px;
        font-weight: 700;
        color: {text_primary} !important;
        margin-bottom: 6px;
        line-height: 1.3;
    }}
    .stripe-banner-desc {{
        font-size: 13px;
        color: {text_secondary} !important;
        line-height: 1.4;
        margin-bottom: 12px;
    }}
    .stripe-banner-link {{
        font-size: 12px;
        font-weight: 600;
        color: {accent_color} !important;
        cursor: pointer;
        display: flex;
        align-items: center;
        gap: 4px;
    }}

    /* Notice Strip */
    .stripe-notice {{
        background-color: {notice_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 8px;
        padding: 10px 16px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 13px;
        color: {text_secondary} !important;
    }}

    /* Utility Info Box */
    .stripe-info-box {{
        background-color: {cell_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 12px;
        font-size: 12px;
        color: {text_secondary} !important;
        line-height: 1.5;
    }}

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {{
        background-color: {surface_color} !important;
        border-right: 1px solid {card_border} !important;
    }}
    section[data-testid="stSidebar"] .block-container {{
        padding-top: 1.8rem;
    }}
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {{
        color: {text_primary} !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 16px;
        margin-bottom: 8px;
    }}
    section[data-testid="stSidebar"] label {{
        color: {text_secondary} !important;
        font-size: 13px !important;
        font-weight: 500 !important;
    }}

    /* Streamlit Native Inputs */
    div[data-baseweb="select"] > div {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
        color: {text_primary} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="select"] span {{
        color: {text_primary} !important;
    }}
    div[data-baseweb="popover"] {{
        background-color: {surface_color} !important;
        color: {text_primary} !important;
    }}
    ul[data-baseweb="menu"] {{
        background-color: {surface_color} !important;
    }}
    li[data-baseweb="menu-item"] {{
        color: {text_primary} !important;
    }}
    li[data-baseweb="menu-item"]:hover {{
        background-color: {surface_hover} !important;
    }}

    /* Date Input */
    div[data-baseweb="input"] > div {{
        background-color: {input_bg} !important;
        border-color: {input_border} !important;
        color: {text_primary} !important;
        border-radius: 8px !important;
    }}
    div[data-baseweb="input"] input {{
        color: {text_primary} !important;
    }}

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 6px;
        background-color: {surface_color} !important;
        padding: 5px;
        border-radius: 10px;
        border: 1px solid {card_border} !important;
        margin-bottom: 20px;
    }}
    .stTabs [data-baseweb="tab"] {{
        height: 36px;
        border-radius: 7px;
        padding: 0 16px;
        background-color: transparent !important;
        border: none !important;
        color: {text_secondary} !important;
        font-size: 13px;
        font-weight: 500;
        transition: all 0.15s ease;
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        color: {text_primary} !important;
        background-color: {surface_hover} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background-color: {accent_color} !important;
        color: #ffffff !important;
        font-weight: 600 !important;
    }}

    /* Streamlit Native Metrics */
    [data-testid="stMetricValue"] {{
        color: {text_primary} !important;
        font-size: 24px !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }}
    [data-testid="stMetricLabel"] {{
        color: {text_secondary} !important;
        font-size: 12px !important;
        font-weight: 500 !important;
    }}

    /* Streamlit Native Expanders */
    [data-testid="stExpander"] {{
        background-color: {card_bg} !important;
        border: 1px solid {card_border} !important;
        border-radius: 10px;
    }}
    [data-testid="stExpander"] summary span {{
        color: {text_primary} !important;
        font-weight: 600;
        font-size: 13px;
    }}
    [data-testid="stExpander"] p, [data-testid="stExpander"] div {{
        color: {text_secondary} !important;
    }}

    /* Tables */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {card_border} !important;
        border-radius: 10px;
        background-color: {card_bg} !important;
    }}
    .stDataFrame table {{
        background: {card_bg} !important;
        color: {text_primary} !important;
    }}
    .stDataFrame th {{
        background: {surface_color} !important;
        color: {text_secondary} !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        border-bottom: 1px solid {card_border} !important;
    }}
    .stDataFrame td {{
        color: {text_primary} !important;
        border-bottom: 1px solid {card_border} !important;
    }}

    /* Plotly Chart Container Visibility Fix */
    .stPlotlyChart {{
        width: 100% !important;
        display: block !important;
        visibility: visible !important;
    }}
    .stPlotlyChart > div {{
        width: 100% !important;
        display: block !important;
        visibility: visible !important;
    }}
    .js-plotly-plot {{
        width: 100% !important;
    }}

    /* Custom Scrollbar */
    ::-webkit-scrollbar {{
        width: 6px;
        height: 6px;
    }}
    ::-webkit-scrollbar-track {{
        background: {bg_color};
    }}
    ::-webkit-scrollbar-thumb {{
        background: {text_muted};
        border-radius: 3px;
        opacity: 0.5;
    }}
    </style>
    """
    return css


def get_plotly_theme(theme="dark"):
    """
    Configures Plotly layout dict matching the Stripe dashboard style in both modes.
    Avoids fixed legend to prevent keyword collisions when customizing individual charts.
    """
    if theme == "dark":
        return {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "Inter, sans-serif", "color": "#94a3b8", "size": 12},
            "xaxis": {
                "gridcolor": "#232636",
                "linecolor": "#232636",
                "zerolinecolor": "#232636",
                "showline": True,
            },
            "yaxis": {
                "gridcolor": "#232636",
                "linecolor": "#232636",
                "zerolinecolor": "#232636",
                "showline": True,
            },
            "hoverlabel": {
                "bgcolor": "#1e2130",
                "bordercolor": "#635bff",
                "font": {"color": "#ffffff", "family": "Inter"},
            },
        }
    else:
        return {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "font": {"family": "Inter, sans-serif", "color": "#475569", "size": 12},
            "xaxis": {
                "gridcolor": "#f1f5f9",
                "linecolor": "#e2e8f0",
                "zerolinecolor": "#e2e8f0",
                "showline": True,
            },
            "yaxis": {
                "gridcolor": "#f1f5f9",
                "linecolor": "#e2e8f0",
                "zerolinecolor": "#e2e8f0",
                "showline": True,
            },
            "hoverlabel": {
                "bgcolor": "#ffffff",
                "bordercolor": "#635bff",
                "font": {"color": "#0f172a", "family": "Inter"},
            },
        }
