"""
report_generator.py
-------------------
Intelligent Report Generation Module for Land Use & Deforestation Analysis.

Responsibilities:
- Synthesize prediction and change detection results into structured findings
- Generate natural-language narrative paragraphs (context-aware, severity-sensitive)
- Produce per-class change summary sentences
- Build Plotly comparison bar charts for dashboard embedding
- Compute a composite Environmental Impact Score (0-100)
- Export full HTML report with embedded charts and findings
- Provide JSON-serializable report dict for API/dashboard use

Output Sentences Examples:
    "Forest cover reduced by 18.4% — significant habitat loss detected."
    "Urban expansion of 9.2% observed, likely encroaching on forest boundaries."
    "Water body area decreased by 3.1%, indicating possible drought stress."
    "Risk classification: HIGH — Immediate environmental intervention required."

Author: AI Land Use Analysis System
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
REPORT_OUTPUT_DIR   = "reports"
REPORT_HTML_PATH    = "reports/land_use_report.html"
REPORT_JSON_PATH    = "reports/land_use_report.json"

# Class display colors for Plotly charts (hex)
CLASS_CHART_COLORS = {
    "Forest"      : "#228B22",
    "Water"       : "#1E90FF",
    "Urban"       : "#DC143C",
    "Agriculture" : "#DAA520",
    "Barren"      : "#D2B48C"
}

# Environmental Impact Score weights (sum = 1.0)
EIS_WEIGHTS = {
    "forest_loss"    : 0.40,   # Highest weight — primary deforestation indicator
    "urban_expansion": 0.25,   # Urban sprawl impact
    "total_change"   : 0.20,   # General landscape disruption
    "water_loss"     : 0.15    # Water body shrinkage
}


# ─────────────────────────────────────────────
# Natural Language Generation
# ─────────────────────────────────────────────

def generate_forest_finding(forest_loss_pct: float,
                             forest_gain_pct: float,
                             old_forest_pct: float,
                             new_forest_pct: float) -> str:
    """
    Generate a context-aware natural-language sentence describing
    forest cover change with severity-calibrated vocabulary.

    Parameters:
        forest_loss_pct (float): % of total area lost to deforestation.
        forest_gain_pct (float): % of total area gained (reforestation).
        old_forest_pct (float): Forest cover in old image (%).
        new_forest_pct (float): Forest cover in new image (%).

    Returns:
        str: Human-readable forest change finding.
    """
    net_change = new_forest_pct - old_forest_pct

    if forest_loss_pct == 0.0 and forest_gain_pct == 0.0:
        return (
            f"Forest cover remained stable at approximately {old_forest_pct:.1f}% "
            f"of the observed area. No significant deforestation detected."
        )

    # Select severity descriptor based on loss magnitude
    if forest_loss_pct < 5.0:
        severity = "minor"
        urgency  = "Continued monitoring is recommended."
    elif forest_loss_pct < 15.0:
        severity = "moderate"
        urgency  = "Local environmental authorities should be notified."
    elif forest_loss_pct < 30.0:
        severity = "significant"
        urgency  = "Urgent investigation and protective measures are required."
    else:
        severity = "critical"
        urgency  = "Emergency conservation intervention is immediately required."

    # Build the primary finding sentence
    finding = (
        f"Forest cover has reduced by {abs(net_change):.1f}% "
        f"(from {old_forest_pct:.1f}% to {new_forest_pct:.1f}% of the observed area), "
        f"representing {severity} deforestation with a loss footprint of "
        f"{forest_loss_pct:.1f}% of total image area. {urgency}"
    )

    # Append reforestation note if significant gain observed
    if forest_gain_pct > 2.0:
        finding += (
            f" However, a reforestation gain of {forest_gain_pct:.1f}% "
            f"was also detected, partially offsetting total forest loss."
        )

    return finding


def generate_urban_finding(urban_expansion_pct: float,
                            old_urban_pct: float,
                            new_urban_pct: float,
                            forest_loss_pct: float) -> str:
    """
    Generate a natural-language finding for urban expansion,
    including correlation with forest loss where relevant.

    Parameters:
        urban_expansion_pct (float): New urban area as % of total area.
        old_urban_pct (float): Urban cover in old image (%).
        new_urban_pct (float): Urban cover in new image (%).
        forest_loss_pct (float): Forest loss % (for correlation analysis).

    Returns:
        str: Human-readable urban expansion finding.
    """
    if urban_expansion_pct < 0.5:
        return (
            f"Urban area remained largely unchanged at {new_urban_pct:.1f}% "
            f"of the observed region. No significant urban expansion detected."
        )

    net_change = new_urban_pct - old_urban_pct

    if urban_expansion_pct < 3.0:
        scale = "limited"
    elif urban_expansion_pct < 8.0:
        scale = "moderate"
    elif urban_expansion_pct < 15.0:
        scale = "substantial"
    else:
        scale = "rapid"

    finding = (
        f"Urban expansion of {net_change:.1f}% detected "
        f"(from {old_urban_pct:.1f}% to {new_urban_pct:.1f}%), "
        f"indicating {scale} growth of built-up infrastructure "
        f"within the observed area."
    )

    # Correlate with forest loss
    if urban_expansion_pct > 2.0 and forest_loss_pct > 5.0:
        finding += (
            f" The concurrent forest loss of {forest_loss_pct:.1f}% suggests "
            f"that urban development may be directly encroaching on forested land."
        )

    return finding


def generate_water_finding(old_water_pct: float,
                            new_water_pct: float,
                            water_change_pct: float) -> str:
    """
    Generate a natural-language finding for water body changes.

    Parameters:
        old_water_pct (float): Water coverage in old image (%).
        new_water_pct (float): Water coverage in new image (%).
        water_change_pct (float): Net change in water coverage (%).

    Returns:
        str: Human-readable water body change finding.
    """
    abs_change = abs(water_change_pct)

    if abs_change < 1.0:
        return (
            f"Water body coverage remained stable at approximately "
            f"{new_water_pct:.1f}% of the observed area."
        )

    direction = "increased" if water_change_pct > 0 else "decreased"
    cause     = "flooding or seasonal inflow" if water_change_pct > 0 else "drought, diversion, or overextraction"

    finding = (
        f"Water body coverage {direction} by {abs_change:.1f}% "
        f"(from {old_water_pct:.1f}% to {new_water_pct:.1f}%), "
        f"potentially indicating {cause} within the region."
    )

    if water_change_pct < -3.0:
        finding += (
            " Significant water body reduction may signal environmental stress "
            "affecting aquatic ecosystems and local water supply."
        )

    return finding


def generate_overall_summary(risk_level: str,
                              risk_description: str,
                              risk_action: str,
                              total_changed_pct: float,
                              num_changed_regions: int,
                              eis_score: float) -> str:
    """
    Generate the executive summary paragraph — the top-level finding
    that captures overall landscape health and urgency.

    Parameters:
        risk_level (str): Low / Medium / High / Critical.
        risk_description (str): Detailed risk description from detect_change.
        risk_action (str): Recommended action string.
        total_changed_pct (float): Total % area with detectable change.
        num_changed_regions (int): Number of spatially distinct changed regions.
        eis_score (float): Environmental Impact Score (0-100).

    Returns:
        str: Executive summary paragraph.
    """
    region_descriptor = (
        "a single concentrated region" if num_changed_regions == 1
        else f"{num_changed_regions} spatially distinct regions"
    )

    eis_descriptor = (
        "minimal"    if eis_score < 25  else
        "moderate"   if eis_score < 50  else
        "severe"     if eis_score < 75  else
        "critical"
    )

    summary = (
        f"Analysis of the submitted satellite imagery pair reveals that "
        f"{total_changed_pct:.1f}% of the observed landscape has undergone "
        f"detectable change, distributed across {region_descriptor}. "
        f"The composite Environmental Impact Score is {eis_score:.1f}/100, "
        f"indicating {eis_descriptor} environmental disruption. "
        f"Risk classification: {risk_level.upper()} — {risk_description} "
        f"Recommended action: {risk_action}"
    )

    return summary


def generate_class_change_bullets(prediction_comparison: dict) -> list:
    """
    Generate a list of bullet-point strings describing per-class land-use
    changes, suitable for dashboard display and HTML report embedding.

    Parameters:
        prediction_comparison (dict): Output of predict.compare_predictions().
            Keys: class names → {old_pct, new_pct, change_pct, direction}

    Returns:
        list: List of formatted finding strings, one per land-use class.
    """
    bullets = []

    change_templates = {
        "Forest": {
            "increased" : "Forest cover expanded by {delta:.1f}% — possible reforestation or seasonal regrowth.",
            "decreased" : "Forest cover reduced by {delta:.1f}% — deforestation or land clearing detected.",
            "unchanged" : "Forest cover remained stable with no significant change detected."
        },
        "Water": {
            "increased" : "Water body area expanded by {delta:.1f}% — possible flooding or seasonal inflow.",
            "decreased" : "Water body area decreased by {delta:.1f}% — possible drought stress or water diversion.",
            "unchanged" : "Water body coverage remained stable throughout the observation period."
        },
        "Urban": {
            "increased" : "Urban area expanded by {delta:.1f}% — infrastructure or settlement growth detected.",
            "decreased" : "Urban area reduced by {delta:.1f}% — possible demolition or land reclamation.",
            "unchanged" : "Urban footprint showed no significant change between observations."
        },
        "Agriculture": {
            "increased" : "Agricultural land expanded by {delta:.1f}% — possible new cultivation or land conversion.",
            "decreased" : "Agricultural land reduced by {delta:.1f}% — possible abandonment or conversion to other use.",
            "unchanged" : "Agricultural land coverage remained consistent across both observations."
        },
        "Barren": {
            "increased" : "Barren land increased by {delta:.1f}% — possible land degradation or post-harvest clearance.",
            "decreased" : "Barren land decreased by {delta:.1f}% — possible revegetation or new construction.",
            "unchanged" : "Barren land area showed no significant change detected."
        }
    }

    for class_name, metrics in prediction_comparison.items():
        delta     = abs(metrics["change_pct"])
        direction = metrics["direction"]
        old_pct   = metrics["old_pct"]
        new_pct   = metrics["new_pct"]

        template_group = change_templates.get(class_name, {
            "increased" : f"{class_name} area increased by {{delta:.1f}}%.",
            "decreased" : f"{class_name} area decreased by {{delta:.1f}}%.",
            "unchanged" : f"{class_name} area showed no significant change."
        })

        sentence = template_group[direction].format(delta=delta)

        # Append before/after context
        sentence += f" ({old_pct:.1f}% → {new_pct:.1f}%)"
        bullets.append(sentence)

    return bullets


# ─────────────────────────────────────────────
# Environmental Impact Score
# ─────────────────────────────────────────────

def compute_environmental_impact_score(forest_loss_pct: float,
                                        urban_expansion_pct: float,
                                        total_changed_pct: float,
                                        water_change_pct: float) -> float:
    """
    Compute a composite Environmental Impact Score (EIS) on a 0-100 scale.

    The EIS combines four damage signals with calibrated weights:
        - Forest loss       : 40% weight (primary deforestation signal)
        - Urban expansion   : 25% weight (infrastructure encroachment)
        - Total change      : 20% weight (overall landscape disruption)
        - Water body loss   : 15% weight (hydrological stress)

    Each component is normalized to [0, 100] before weighting.
    Higher score = greater environmental impact.

    Parameters:
        forest_loss_pct (float): % forest area lost.
        urban_expansion_pct (float): % new urban area.
        total_changed_pct (float): % total changed area.
        water_change_pct (float): % water change (negative = loss).

    Returns:
        float: EIS score [0.0, 100.0].
    """
    # Normalize each component: cap at plausible max values
    forest_component  = min(forest_loss_pct / 60.0,  1.0) * 100
    urban_component   = min(urban_expansion_pct / 30.0, 1.0) * 100
    change_component  = min(total_changed_pct / 80.0, 1.0) * 100
    # Water loss only (positive change = not harmful)
    water_loss        = max(0.0, -water_change_pct)
    water_component   = min(water_loss / 20.0, 1.0) * 100

    eis = (
        EIS_WEIGHTS["forest_loss"]     * forest_component  +
        EIS_WEIGHTS["urban_expansion"] * urban_component   +
        EIS_WEIGHTS["total_change"]    * change_component  +
        EIS_WEIGHTS["water_loss"]      * water_component
    )

    eis = round(min(max(eis, 0.0), 100.0), 1)

    logger.info(
        f"EIS Score: {eis:.1f}/100 | "
        f"Forest={forest_component:.1f}, Urban={urban_component:.1f}, "
        f"Change={change_component:.1f}, Water={water_component:.1f}"
    )
    return eis


# ─────────────────────────────────────────────
# Plotly Chart Builders
# ─────────────────────────────────────────────

def build_land_use_comparison_chart(old_area_stats: pd.DataFrame,
                                     new_area_stats: pd.DataFrame) -> go.Figure:
    """
    Build a grouped bar chart comparing land-use class percentages
    between the old and new satellite images.

    Parameters:
        old_area_stats (pd.DataFrame): Area stats from old image prediction.
        new_area_stats (pd.DataFrame): Area stats from new image prediction.

    Returns:
        go.Figure: Plotly grouped bar chart figure.
    """
    old_dict = old_area_stats.set_index("Class")["Percentage"].to_dict()
    new_dict = new_area_stats.set_index("Class")["Percentage"].to_dict()
    classes  = list(CLASS_CHART_COLORS.keys())

    old_vals = [old_dict.get(c, 0.0) for c in classes]
    new_vals = [new_dict.get(c, 0.0) for c in classes]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Before (Old Image)",
        x=classes,
        y=old_vals,
        marker_color=[CLASS_CHART_COLORS[c] for c in classes],
        marker_opacity=0.55,
        marker_line_color=[CLASS_CHART_COLORS[c] for c in classes],
        marker_line_width=2,
        text=[f"{v:.1f}%" for v in old_vals],
        textposition="outside"
    ))

    fig.add_trace(go.Bar(
        name="After (New Image)",
        x=classes,
        y=new_vals,
        marker_color=[CLASS_CHART_COLORS[c] for c in classes],
        marker_opacity=1.0,
        marker_line_color="white",
        marker_line_width=1.5,
        text=[f"{v:.1f}%" for v in new_vals],
        textposition="outside"
    ))

    fig.update_layout(
        title=dict(
            text="Land Use Coverage: Before vs After",
            font=dict(size=16, color="#2C3E50"),
            x=0.5
        ),
        barmode="group",
        xaxis_title="Land Use Class",
        yaxis_title="Coverage (%)",
        yaxis=dict(range=[0, max(max(old_vals), max(new_vals)) * 1.25]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(t=80, b=50, l=60, r=30)
    )

    return fig


def build_forest_change_gauge(forest_loss_pct: float,
                               eis_score: float) -> go.Figure:
    """
    Build a dual-gauge figure showing:
    - Left gauge: Forest loss percentage (0-100%)
    - Right gauge: Environmental Impact Score (0-100)

    Parameters:
        forest_loss_pct (float): Detected forest loss percentage.
        eis_score (float): Computed Environmental Impact Score.

    Returns:
        go.Figure: Plotly indicator gauge figure.
    """
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]]
    )

    # Forest loss gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=forest_loss_pct,
        title={"text": "Forest Loss (%)", "font": {"size": 14}},
        gauge={
            "axis"      : {"range": [0, 100]},
            "bar"       : {"color": "#E74C3C"},
            "steps"     : [
                {"range": [0,  10],  "color": "#EAFAF1"},
                {"range": [10, 25],  "color": "#FDEBD0"},
                {"range": [25, 50],  "color": "#FADBD8"},
                {"range": [50, 100], "color": "#922B21"}
            ],
            "threshold" : {
                "line" : {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": forest_loss_pct
            }
        },
        number={"suffix": "%", "font": {"size": 28}},
        domain={"row": 0, "column": 0}
    ), row=1, col=1)

    # EIS gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=eis_score,
        title={"text": "Environmental Impact Score", "font": {"size": 14}},
        gauge={
            "axis"  : {"range": [0, 100]},
            "bar"   : {"color": "#8E44AD"},
            "steps" : [
                {"range": [0,  25],  "color": "#EAFAF1"},
                {"range": [25, 50],  "color": "#FDEBD0"},
                {"range": [50, 75],  "color": "#FADBD8"},
                {"range": [75, 100], "color": "#922B21"}
            ]
        },
        number={"suffix": "/100", "font": {"size": 28}},
        domain={"row": 0, "column": 1}
    ), row=1, col=2)

    fig.update_layout(
        title=dict(
            text="Key Environmental Indicators",
            font=dict(size=16, color="#2C3E50"),
            x=0.5
        ),
        height=320,
        paper_bgcolor="white",
        margin=dict(t=60, b=20, l=30, r=30)
    )

    return fig


def build_change_breakdown_pie(change_results: dict) -> go.Figure:
    """
    Build a pie chart showing the breakdown of detected changes:
    forest loss, urban expansion, water change, and other changes.

    Parameters:
        change_results (dict): Output from detect_change.run_change_detection().

    Returns:
        go.Figure: Plotly pie chart.
    """
    labels = ["Forest Loss", "Urban Expansion", "Water Change", "Other Change"]
    values = [
        max(change_results["forest_loss_pct"], 0.0),
        max(change_results["urban_expansion_pct"], 0.0),
        max(abs(change_results["water_change_pct"]), 0.0),
        max(
            change_results["total_changed_pct"]
            - change_results["forest_loss_pct"]
            - change_results["urban_expansion_pct"]
            - abs(change_results["water_change_pct"]),
            0.0
        )
    ]

    # Remove zero entries
    filtered = [(l, v) for l, v in zip(labels, values) if v > 0]
    if not filtered:
        filtered = [("No Significant Change", 1.0)]

    labels_f, values_f = zip(*filtered)
    colors = ["#E74C3C", "#E67E22", "#3498DB", "#95A5A6"][:len(labels_f)]

    fig = go.Figure(go.Pie(
        labels=labels_f,
        values=values_f,
        hole=0.42,
        marker=dict(colors=colors, line=dict(color="white", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12),
        hovertemplate="<b>%{label}</b><br>%{value:.2f}% of total area<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Change Composition Breakdown",
            font=dict(size=16, color="#2C3E50"),
            x=0.5
        ),
        paper_bgcolor="white",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        margin=dict(t=60, b=60, l=20, r=20),
        height=380
    )

    return fig


def build_class_trend_chart(prediction_comparison: dict) -> go.Figure:
    """
    Build a horizontal bar chart showing net change per land-use class,
    colour-coded green for gain and red for loss.

    Parameters:
        prediction_comparison (dict): Output of predict.compare_predictions().

    Returns:
        go.Figure: Plotly horizontal bar chart.
    """
    classes = list(prediction_comparison.keys())
    changes = [prediction_comparison[c]["change_pct"] for c in classes]
    colors  = ["#27AE60" if v >= 0 else "#E74C3C" for v in changes]

    fig = go.Figure(go.Bar(
        x=changes,
        y=classes,
        orientation="h",
        marker_color=colors,
        marker_line_color="white",
        marker_line_width=1,
        text=[f"{v:+.2f}%" for v in changes],
        textposition="outside"
    ))

    fig.add_vline(x=0, line_width=2, line_color="#2C3E50")

    fig.update_layout(
        title=dict(
            text="Net Land Use Change per Class (Before → After)",
            font=dict(size=16, color="#2C3E50"),
            x=0.5
        ),
        xaxis_title="Change (%)",
        yaxis_title="Land Use Class",
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="white",
        font=dict(family="Arial", size=12),
        margin=dict(t=70, b=50, l=120, r=60),
        height=320
    )

    return fig


# ─────────────────────────────────────────────
# HTML Report Builder
# ─────────────────────────────────────────────

def build_html_report(report_data: dict) -> str:
    """
    Build a complete self-contained HTML report embedding all findings,
    Plotly charts (as inline JS), and metadata.

    Parameters:
        report_data (dict): Full structured report dict from generate_full_report().

    Returns:
        str: Complete HTML document as a string.
    """
    timestamp    = report_data["metadata"]["generated_at"]
    risk_level   = report_data["risk"]["level"]
    risk_color   = report_data["risk"]["color"]
    eis_score    = report_data["eis_score"]
    summary      = report_data["findings"]["summary"]
    forest_f     = report_data["findings"]["forest"]
    urban_f      = report_data["findings"]["urban"]
    water_f      = report_data["findings"]["water"]
    bullets      = report_data["findings"]["class_bullets"]

    # Serialize Plotly charts to HTML divs
    comparison_chart_html = report_data["charts"]["comparison_chart"].to_html(
        full_html=False, include_plotlyjs="cdn"
    )
    gauge_chart_html = report_data["charts"]["gauge_chart"].to_html(
        full_html=False, include_plotlyjs=False
    )
    pie_chart_html = report_data["charts"]["pie_chart"].to_html(
        full_html=False, include_plotlyjs=False
    )
    trend_chart_html = report_data["charts"]["trend_chart"].to_html(
        full_html=False, include_plotlyjs=False
    )

    bullets_html = "".join(
        f"<li style='margin-bottom:8px'>{b}</li>" for b in bullets
    )

    # Risk badge style
    risk_badge = (
        f'<span style="background:{risk_color};color:white;'
        f'padding:6px 16px;border-radius:20px;font-weight:bold;'
        f'font-size:1.1em;">{risk_level.upper()}</span>'
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
    <title>Land Use & Deforestation Analysis Report</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            background: #F0F4F8;
            color: #2C3E50;
            line-height: 1.7;
        }}
        .container {{ max-width: 1100px; margin: 0 auto; padding: 24px; }}
        .header {{
            background: linear-gradient(135deg, #1B4F72, #2E86C1);
            color: white;
            padding: 36px 40px;
            border-radius: 12px;
            margin-bottom: 28px;
            box-shadow: 0 4px 18px rgba(0,0,0,0.15);
        }}
        .header h1 {{ font-size: 2em; font-weight: 700; margin-bottom: 6px; }}
        .header p  {{ opacity: 0.85; font-size: 0.95em; }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 28px 32px;
            margin-bottom: 22px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.07);
        }}
        .card h2 {{
            font-size: 1.25em;
            color: #1B4F72;
            border-bottom: 2px solid #EBF5FB;
            padding-bottom: 10px;
            margin-bottom: 18px;
        }}
        .summary-box {{
            background: #EBF5FB;
            border-left: 5px solid #2E86C1;
            padding: 18px 22px;
            border-radius: 0 8px 8px 0;
            font-size: 1.02em;
            margin-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin-bottom: 18px;
        }}
        .metric-box {{
            background: #F8F9FA;
            border: 1px solid #DEE2E6;
            border-radius: 8px;
            padding: 16px 18px;
            text-align: center;
        }}
        .metric-box .value {{
            font-size: 1.9em;
            font-weight: 700;
            color: #1B4F72;
        }}
        .metric-box .label {{
            font-size: 0.8em;
            color: #7F8C8D;
            margin-top: 4px;
        }}
        .finding-block {{
            background: #FDFEFE;
            border: 1px solid #E8EAED;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 12px;
            font-size: 0.97em;
        }}
        .finding-block strong {{ color: #1B4F72; }}
        ul.findings-list {{
            padding-left: 20px;
            margin-top: 6px;
        }}
        .chart-section {{ margin-top: 8px; }}
        .footer {{
            text-align: center;
            color: #95A5A6;
            font-size: 0.82em;
            padding: 20px 0 10px;
        }}
    </style>
</head>
<body>
<div class="container">

    <!-- Header -->
    <div class="header">
        <h1>Land Use &amp; Deforestation Analysis Report</h1>
        <p>AI-Powered Satellite Imagery Analysis &nbsp;|&nbsp; Generated: {timestamp}</p>
        <p style="margin-top:12px">Risk Classification: {risk_badge}
           &nbsp;&nbsp; EIS Score: <strong>{eis_score:.1f} / 100</strong>
        </p>
    </div>

    <!-- Key Metrics -->
    <div class="card">
        <h2>Key Metrics</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="value">{report_data['metrics']['forest_loss_pct']:.1f}%</div>
                <div class="label">Forest Loss</div>
            </div>
            <div class="metric-box">
                <div class="value">{report_data['metrics']['forest_gain_pct']:.1f}%</div>
                <div class="label">Forest Gain</div>
            </div>
            <div class="metric-box">
                <div class="value">{report_data['metrics']['urban_expansion_pct']:.1f}%</div>
                <div class="label">Urban Expansion</div>
            </div>
            <div class="metric-box">
                <div class="value">{report_data['metrics']['water_change_pct']:+.1f}%</div>
                <div class="label">Water Change</div>
            </div>
            <div class="metric-box">
                <div class="value">{report_data['metrics']['total_changed_pct']:.1f}%</div>
                <div class="label">Total Changed Area</div>
            </div>
            <div class="metric-box">
                <div class="value">{report_data['metrics']['num_changed_regions']}</div>
                <div class="label">Changed Regions</div>
            </div>
        </div>
    </div>

    <!-- Executive Summary -->
    <div class="card">
        <h2>Executive Summary</h2>
        <div class="summary-box">{summary}</div>
    </div>

    <!-- Detailed Findings -->
    <div class="card">
        <h2>Detailed Findings</h2>
        <div class="finding-block"><strong>Forest:</strong> {forest_f}</div>
        <div class="finding-block"><strong>Urban:</strong> {urban_f}</div>
        <div class="finding-block"><strong>Water:</strong> {water_f}</div>

        <h2 style="margin-top:20px">Per-Class Land Use Changes</h2>
        <ul class="findings-list">
            {bullets_html}
        </ul>
    </div>

    <!-- Charts -->
    <div class="card chart-section">
        <h2>Comparative Coverage Analysis</h2>
        {comparison_chart_html}
    </div>

    <div class="card chart-section">
        <h2>Environmental Indicators</h2>
        {gauge_chart_html}
    </div>

    <div class="card chart-section" style="display:grid;grid-template-columns:1fr 1fr;gap:20px">
        <div>
            <h2>Change Composition</h2>
            {pie_chart_html}
        </div>
        <div>
            <h2>Net Class Change</h2>
            {trend_chart_html}
        </div>
    </div>

    <!-- Risk & Action -->
    <div class="card">
        <h2>Risk Assessment &amp; Recommended Action</h2>
        <div class="finding-block">
            <strong>Risk Level:</strong>
            <span style="color:{risk_color};font-weight:bold">{risk_level.upper()}</span>
        </div>
        <div class="finding-block">
            <strong>Description:</strong> {report_data['risk']['description']}
        </div>
        <div class="finding-block">
            <strong>Recommended Action:</strong> {report_data['risk']['action']}
        </div>
    </div>

    <div class="footer">
        AI-Powered Land Use &amp; Deforestation Analysis System &nbsp;|&nbsp;
        Report generated automatically from satellite imagery pair.
    </div>
</div>
</body>
</html>"""

    return html


# ─────────────────────────────────────────────
# Master Report Generator
# ─────────────────────────────────────────────

def generate_full_report(change_results: dict,
                          old_area_stats: pd.DataFrame,
                          new_area_stats: pd.DataFrame,
                          prediction_comparison: dict,
                          save_html: bool = True,
                          save_json: bool = True) -> dict:
    """
    Master report generation pipeline. Assembles all analysis outputs
    into a single structured report dict, builds Plotly charts,
    generates natural-language findings, and optionally exports HTML/JSON.

    Parameters:
        change_results (dict): Output from detect_change.run_change_detection().
        old_area_stats (pd.DataFrame): Area stats from old image prediction.
        new_area_stats (pd.DataFrame): Area stats from new image prediction.
        prediction_comparison (dict): Output from predict.compare_predictions().
        save_html (bool): If True, export HTML report to disk.
        save_json (bool): If True, export JSON report to disk.

    Returns:
        dict: Complete report data with findings, charts, metrics, and metadata.
    """
    logger.info("=" * 60)
    logger.info("GENERATING ANALYSIS REPORT")
    logger.info("=" * 60)

    os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── Extract core metrics ──
    metrics = {
        "forest_loss_pct"       : change_results["forest_loss_pct"],
        "forest_gain_pct"       : change_results["forest_gain_pct"],
        "net_forest_change_pct" : change_results["net_forest_change_pct"],
        "old_forest_pct"        : change_results["old_forest_pct"],
        "new_forest_pct"        : change_results["new_forest_pct"],
        "urban_expansion_pct"   : change_results["urban_expansion_pct"],
        "old_urban_pct"         : change_results["old_urban_pct"],
        "new_urban_pct"         : change_results["new_urban_pct"],
        "water_change_pct"      : change_results["water_change_pct"],
        "old_water_pct"         : change_results["old_water_pct"],
        "new_water_pct"         : change_results["new_water_pct"],
        "total_changed_pct"     : change_results["total_changed_pct"],
        "num_changed_regions"   : change_results["num_changed_regions"]
    }

    # ── Environmental Impact Score ──
    eis_score = compute_environmental_impact_score(
        forest_loss_pct    = metrics["forest_loss_pct"],
        urban_expansion_pct= metrics["urban_expansion_pct"],
        total_changed_pct  = metrics["total_changed_pct"],
        water_change_pct   = metrics["water_change_pct"]
    )

    # ── Natural Language Findings ──
    forest_finding = generate_forest_finding(
        metrics["forest_loss_pct"], metrics["forest_gain_pct"],
        metrics["old_forest_pct"], metrics["new_forest_pct"]
    )
    urban_finding = generate_urban_finding(
        metrics["urban_expansion_pct"], metrics["old_urban_pct"],
        metrics["new_urban_pct"], metrics["forest_loss_pct"]
    )
    water_finding = generate_water_finding(
        metrics["old_water_pct"], metrics["new_water_pct"],
        metrics["water_change_pct"]
    )
    class_bullets = generate_class_change_bullets(prediction_comparison)
    summary = generate_overall_summary(
        risk_level          = change_results["risk_level"],
        risk_description    = change_results["risk_description"],
        risk_action         = change_results["risk_action"],
        total_changed_pct   = metrics["total_changed_pct"],
        num_changed_regions = metrics["num_changed_regions"],
        eis_score           = eis_score
    )

    # ── Charts ──
    comparison_chart = build_land_use_comparison_chart(old_area_stats, new_area_stats)
    gauge_chart      = build_forest_change_gauge(metrics["forest_loss_pct"], eis_score)
    pie_chart        = build_change_breakdown_pie(change_results)
    trend_chart      = build_class_trend_chart(prediction_comparison)

    logger.info("All charts built.")

    # ── Assemble Report Dict ──
    report = {
        "metadata": {
            "generated_at"  : timestamp,
            "system"        : "AI Land Use & Deforestation Analysis v1.0"
        },
        "metrics"  : metrics,
        "eis_score": eis_score,
        "risk"     : {
            "level"      : change_results["risk_level"],
            "color"      : change_results["risk_color"],
            "description": change_results["risk_description"],
            "action"     : change_results["risk_action"]
        },
        "findings" : {
            "summary"      : summary,
            "forest"       : forest_finding,
            "urban"        : urban_finding,
            "water"        : water_finding,
            "class_bullets": class_bullets
        },
        "charts"   : {
            "comparison_chart" : comparison_chart,
            "gauge_chart"      : gauge_chart,
            "pie_chart"        : pie_chart,
            "trend_chart"      : trend_chart
        }
    }

    # ── Export HTML ──
    if save_html:
        html_content = build_html_report(report)
        with open(REPORT_HTML_PATH, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"HTML report saved: {REPORT_HTML_PATH}")

    # ── Export JSON (charts excluded — not serializable) ──
    if save_json:
        json_safe = {
            k: v for k, v in report.items() if k != "charts"
        }
        with open(REPORT_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(json_safe, f, indent=2)
        logger.info(f"JSON report saved: {REPORT_JSON_PATH}")

    logger.info("Report generation complete.")
    logger.info(f"  EIS Score   : {eis_score:.1f}/100")
    logger.info(f"  Risk Level  : {change_results['risk_level']}")
    logger.info(f"  Summary     : {summary[:80]}...")

    return report


# ─────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    """
    Standalone test with synthetic data.
    In production this is called from app.py with real pipeline outputs.
    """
    print("\n Running report generator with synthetic test data...\n")

    # Synthetic change results
    synthetic_change = {
        "forest_loss_pct"       : 18.4,
        "forest_gain_pct"       : 2.1,
        "net_forest_change_pct" : -16.3,
        "old_forest_pct"        : 52.0,
        "new_forest_pct"        : 35.7,
        "urban_expansion_pct"   : 9.2,
        "old_urban_pct"         : 12.0,
        "new_urban_pct"         : 21.2,
        "water_change_pct"      : -3.1,
        "old_water_pct"         : 8.5,
        "new_water_pct"         : 5.4,
        "total_changed_pct"     : 34.2,
        "num_changed_regions"   : 7,
        "risk_level"            : "High",
        "risk_color"            : "#E74C3C",
        "risk_description"      : "Severe deforestation detected.",
        "risk_action"           : "Immediate environmental intervention required."
    }

    # Synthetic area stats
    classes = ["Forest", "Water", "Urban", "Agriculture", "Barren"]
    old_stats = pd.DataFrame({
        "Class"      : classes,
        "Pixel_Count": [13312, 2176, 3072, 5120, 2560],
        "Percentage" : [52.0,  8.5,  12.0, 20.0, 7.5]
    })
    new_stats = pd.DataFrame({
        "Class"      : classes,
        "Pixel_Count": [9139, 1382, 5427, 5427, 3225],
        "Percentage" : [35.7, 5.4,  21.2, 21.2, 12.6]
    })

    # Synthetic prediction comparison
    comparison = {
        "Forest"     : {"old_pct": 52.0, "new_pct": 35.7, "change_pct": -16.3, "direction": "decreased"},
        "Water"      : {"old_pct": 8.5,  "new_pct": 5.4,  "change_pct": -3.1,  "direction": "decreased"},
        "Urban"      : {"old_pct": 12.0, "new_pct": 21.2, "change_pct": 9.2,   "direction": "increased"},
        "Agriculture": {"old_pct": 20.0, "new_pct": 21.2, "change_pct": 1.2,   "direction": "increased"},
        "Barren"     : {"old_pct": 7.5,  "new_pct": 12.6, "change_pct": 5.1,   "direction": "increased"}
    }

    report = generate_full_report(
        change_results        = synthetic_change,
        old_area_stats        = old_stats,
        new_area_stats        = new_stats,
        prediction_comparison = comparison,
        save_html             = True,
        save_json             = True
    )

    print(f"  EIS Score  : {report['eis_score']:.1f}/100")
    print(f"  Risk Level : {report['risk']['level']}")
    print(f"\n  Summary:\n  {report['findings']['summary']}\n")
    print(f"  Forest Finding:\n  {report['findings']['forest']}\n")
    print(f"  Urban Finding:\n  {report['findings']['urban']}\n")
    print(f"  Water Finding:\n  {report['findings']['water']}\n")
    print(f"  HTML report: {REPORT_HTML_PATH}")
    print(f"  JSON report: {REPORT_JSON_PATH}")