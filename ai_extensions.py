"""
ai_extensions.py
----------------
Advanced AI Extensions for Land Use & Deforestation Analysis.

Modules inside this file:
    1. CarbonEstimator      — AI carbon stock & CO2 emission calculator
    2. TrendPredictor       — Forest cover trend forecast with future projection
    3. ImageCaptioner       — Gemini Vision direct satellite image analysis
    4. LocationIntelligence — Region-aware contextual analysis via coordinates
    5. SpeciesImpactAI      — Biodiversity & species-at-risk estimator
    6. SmartAlertSystem     — AI-driven severity-aware alert generator
    7. AnomalyDetector      — Statistical anomaly detection in change patterns

All modules degrade gracefully if Gemini API key is not set.

Author: AI Land Use Analysis System
"""

import os
import io
import base64
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import google.generativeai as genai
from PIL import Image

load_dotenv()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL    = "models/gemini-2.5-flash"
GEMINI_VISION   = "models/gemini-2.5-flash"   # supports vision input


def _get_model():
    """Return a configured Gemini GenerativeModel instance."""
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "GEMINI_API_KEY not set. "
            "Get a free key at https://aistudio.google.com/app/apikey"
        )
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL)


def _get_vision_model():
    """Return a Gemini vision-capable model instance."""
    if not GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY not set.")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_VISION)


# ══════════════════════════════════════════════════════════════
# 1. CARBON STOCK ESTIMATOR
# ══════════════════════════════════════════════════════════════

class CarbonEstimator:
    """
    AI-powered carbon stock loss estimator.

    Uses Gemini to reason about carbon density based on:
    - Forest cover percentage and loss
    - Region type (tropical, temperate, boreal)
    - Estimated area in hectares (derived from image scale)

    Outputs:
    - Estimated tonnes of carbon lost
    - CO2 equivalent emissions
    - Comparison to real-world equivalents (cars, flights, etc.)
    """

    # Average above-ground carbon density by forest type (tC/ha)
    CARBON_DENSITY = {
        "tropical"  : 180,   # Amazon, Congo basin
        "temperate" : 100,   # Europe, North America
        "boreal"    : 60,    # Siberia, Canada
        "unknown"   : 120    # Global average
    }

    def estimate_carbon_loss(self,
                              forest_loss_pct: float,
                              image_area_ha: float = 10000,
                              forest_type: str = "unknown") -> dict:
        """
        Estimate carbon stock lost from detected deforestation.

        Parameters:
            forest_loss_pct (float): % of total area that lost forest cover.
            image_area_ha (float): Estimated real-world area of image in hectares.
                                   Default 10,000 ha = 100 km² (typical satellite tile).
            forest_type (str): tropical / temperate / boreal / unknown.

        Returns:
            dict: Carbon loss metrics and CO2 equivalents.
        """
        density       = self.CARBON_DENSITY.get(forest_type, 120)
        lost_area_ha  = (forest_loss_pct / 100) * image_area_ha
        carbon_tonnes = lost_area_ha * density
        co2_tonnes    = carbon_tonnes * 3.67  # Carbon to CO2 conversion factor

        # Real-world equivalents
        cars_per_year     = co2_tonnes / 4.6        # Average car = 4.6 tCO2/year
        flights_london_ny = co2_tonnes / 0.986      # LHR-JFK = ~0.986 tCO2/person
        homes_powered     = co2_tonnes / 7.5        # Average US home = 7.5 tCO2/year

        result = {
            "forest_type"       : forest_type,
            "lost_area_ha"      : round(lost_area_ha, 1),
            "carbon_tonnes"     : round(carbon_tonnes, 1),
            "co2_tonnes"        : round(co2_tonnes, 1),
            "equivalent_cars"   : round(cars_per_year),
            "equivalent_flights": round(flights_london_ny),
            "equivalent_homes"  : round(homes_powered)
        }

        logger.info(f"Carbon estimate: {carbon_tonnes:.0f} tC lost "
                    f"({co2_tonnes:.0f} tCO2 equivalent)")
        return result

    def generate_ai_carbon_report(self,
                                   carbon_data: dict,
                                   change_results: dict) -> str:
        """
        Use Gemini to generate a contextual carbon impact narrative
        that goes beyond the raw numbers.

        Parameters:
            carbon_data (dict): Output of estimate_carbon_loss().
            change_results (dict): Full change detection results.

        Returns:
            str: AI-generated carbon impact analysis.
        """
        prompt = f"""
You are a carbon accounting expert and climate scientist.

DEFORESTATION DATA:
  Forest loss area    : {carbon_data['lost_area_ha']:.0f} hectares
  Carbon stock lost   : {carbon_data['carbon_tonnes']:.0f} tonnes of carbon
  CO2 equivalent      : {carbon_data['co2_tonnes']:.0f} tonnes of CO2
  Forest type         : {carbon_data['forest_type']}
  Equivalent to       : {carbon_data['equivalent_cars']:,} cars driven for a year
                        {carbon_data['equivalent_flights']:,} London-New York flights
                        {carbon_data['equivalent_homes']:,} homes powered for a year

  Additional context  : Forest loss {change_results['forest_loss_pct']:.1f}%,
                        Urban expansion {change_results['urban_expansion_pct']:.1f}%,
                        Risk level: {change_results['risk_level']}

Write a concise 3-4 sentence carbon impact statement that:
1. Puts the carbon loss in meaningful real-world perspective
2. Explains the climate significance of this loss
3. Mentions Paris Agreement / net-zero implications if relevant
4. Suggests one specific carbon-offset or mitigation action

Be specific and impactful. No generic filler.
"""
        try:
            model    = _get_model()
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return (f"Carbon loss estimated at {carbon_data['co2_tonnes']:.0f} "
                    f"tonnes CO2 equivalent — comparable to "
                    f"{carbon_data['equivalent_cars']:,} cars driven for one year.")


# ══════════════════════════════════════════════════════════════
# 2. TREND PREDICTOR
# ══════════════════════════════════════════════════════════════

class TrendPredictor:
    """
    Forest cover trend forecaster using linear regression
    combined with Gemini narrative for context-aware predictions.

    Given current and past forest cover percentages,
    projects future trajectory and time-to-critical-threshold.
    """

    def predict_forest_trajectory(self,
                                   old_forest_pct: float,
                                   new_forest_pct: float,
                                   years_between: float = 2.0,
                                   forecast_years: int = 10) -> dict:
        """
        Project future forest cover using linear trend extrapolation.

        Parameters:
            old_forest_pct (float): Forest cover in old image (%).
            new_forest_pct (float): Forest cover in new image (%).
            years_between (float): Years between the two image captures.
            forecast_years (int): How many years ahead to forecast.

        Returns:
            dict: Trajectory data including annual rate and future projections.
        """
        annual_change = (new_forest_pct - old_forest_pct) / years_between
        current_year  = datetime.now().year

        # Project forward
        projections = []
        for yr in range(0, forecast_years + 1):
            projected_pct = max(0.0, new_forest_pct + (annual_change * yr))
            projections.append({
                "year"          : current_year + yr,
                "forest_pct"    : round(projected_pct, 2)
            })

        # Time to reach critical thresholds
        def years_to_threshold(threshold_pct):
            if annual_change >= 0:
                return None   # Not declining
            return round((new_forest_pct - threshold_pct) / abs(annual_change), 1)

        result = {
            "annual_change_pct"     : round(annual_change, 3),
            "current_forest_pct"    : new_forest_pct,
            "projections"           : projections,
            "years_to_50pct_loss"   : years_to_threshold(new_forest_pct * 0.5),
            "years_to_complete_loss": years_to_threshold(0.0),
            "years_to_critical_10"  : years_to_threshold(10.0),
            "trend_direction"       : (
                "improving"  if annual_change > 0.5 else
                "stable"     if -0.5 <= annual_change <= 0.5 else
                "declining"
            )
        }

        logger.info(f"Trend: {annual_change:+.2f}%/year | "
                    f"Direction: {result['trend_direction']}")
        return result

    def generate_ai_forecast_narrative(self,
                                        trend_data: dict,
                                        change_results: dict) -> str:
        """
        Use Gemini to generate an intelligent forecast narrative
        explaining what the trend means ecologically and politically.

        Parameters:
            trend_data (dict): Output of predict_forest_trajectory().
            change_results (dict): Full change detection results.

        Returns:
            str: AI-generated forecast narrative.
        """
        years_critical = trend_data.get("years_to_critical_10", "N/A")
        years_loss     = trend_data.get("years_to_complete_loss", "N/A")

        prompt = f"""
You are an environmental forecasting expert.

FOREST TREND DATA:
  Current forest cover      : {trend_data['current_forest_pct']:.1f}%
  Annual change rate        : {trend_data['annual_change_pct']:+.3f}% per year
  Trend direction           : {trend_data['trend_direction']}
  Years to reach 10% cover  : {years_critical}
  Years to complete loss    : {years_loss}
  Current risk level        : {change_results['risk_level']}
  Urban expansion rate      : {change_results['urban_expansion_pct']:.1f}%

Write a 3-4 sentence future forecast that:
1. States clearly what will happen to this forest at the current rate
2. Identifies the most critical timeframe for intervention
3. Explains what ecological tipping points may be crossed
4. Gives one specific recommendation to reverse this trend

Be direct. Use the actual numbers. No vague language.
"""
        try:
            model    = _get_model()
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            rate = trend_data['annual_change_pct']
            return (
                f"At the current rate of {rate:+.2f}% per year, "
                f"forest cover is projected to reach critical levels "
                f"within {years_critical} years without intervention."
            )

    def build_forecast_chart_data(self, trend_data: dict) -> pd.DataFrame:
        """
        Convert projection data to a DataFrame suitable for Plotly charts.

        Parameters:
            trend_data (dict): Output of predict_forest_trajectory().

        Returns:
            pd.DataFrame: Columns — Year, Forest_Pct, Type (Actual/Projected).
        """
        rows = []
        for proj in trend_data["projections"]:
            label = "Projected"
            rows.append({
                "Year"       : proj["year"],
                "Forest_Pct" : proj["forest_pct"],
                "Type"       : label
            })
        return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 3. GEMINI VISION IMAGE CAPTIONER
# ══════════════════════════════════════════════════════════════

class ImageCaptioner:
    """
    Uses Gemini Vision to directly analyse satellite images
    without needing a trained TensorFlow model.

    Gemini Vision can:
    - Describe what it sees in the landscape
    - Identify land use types from the image directly
    - Spot unusual patterns, deforestation edges, water bodies
    - Compare two images and describe changes
    """

    def _image_to_base64_part(self, image_source) -> dict:
        """
        Convert an image (bytes, path, or numpy array) to a
        Gemini-compatible inline image part.

        Parameters:
            image_source: bytes, file path string, or numpy uint8 array.

        Returns:
            dict: Gemini inline_data part dict.
        """
        if isinstance(image_source, np.ndarray):
            img = Image.fromarray(image_source.astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            image_bytes = buf.getvalue()

        elif isinstance(image_source, bytes):
            image_bytes = image_source

        else:
            with open(str(image_source), "rb") as f:
                image_bytes = f.read()

        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return {
            "inline_data": {
                "mime_type": "image/jpeg",
                "data"     : encoded
            }
        }

    def caption_single_image(self, image_source) -> str:
        """
        Generate an intelligent caption for a single satellite image
        by sending it directly to Gemini Vision.

        Parameters:
            image_source: Raw image (bytes, path, or numpy array).

        Returns:
            str: Gemini's landscape description and land use analysis.
        """
        try:
            model      = _get_vision_model()
            image_part = self._image_to_base64_part(image_source)

            prompt = """
You are a remote sensing and satellite imagery expert.
Analyse this satellite image and provide:

1. LANDSCAPE DESCRIPTION (2 sentences)
   What does this area look like from above?

2. LAND USE IDENTIFICATION
   List the visible land use types and their approximate coverage:
   - Forest/Vegetation: X%
   - Water bodies: X%
   - Urban/Built-up: X%
   - Agriculture: X%
   - Barren/Other: X%

3. NOTABLE FEATURES (2-3 bullet points)
   Any unusual patterns, boundaries, deforestation edges,
   river systems, roads, or agricultural patterns visible.

4. OVERALL ASSESSMENT (1 sentence)
   The general environmental health of this area.

Be specific about what you actually see. Do not guess.
"""
            response = model.generate_content([prompt, image_part])
            logger.info("Single image caption generated via Gemini Vision.")
            return response.text

        except Exception as e:
            logger.error(f"Image captioning failed: {e}")
            return f"Image analysis unavailable: {str(e)}"

    def compare_two_images(self,
                            old_image_source,
                            new_image_source) -> str:
        """
        Send BOTH satellite images to Gemini Vision simultaneously
        and ask it to directly describe what changed — no OpenCV needed.

        Parameters:
            old_image_source: Old satellite image.
            new_image_source: New satellite image.

        Returns:
            str: Gemini's visual comparison analysis.
        """
        try:
            model     = _get_vision_model()
            old_part  = self._image_to_base64_part(old_image_source)
            new_part  = self._image_to_base64_part(new_image_source)

            prompt = """
You are a satellite imagery change detection expert.
You are shown TWO images of the same geographic area taken at different times.
The FIRST image is the OLDER capture. The SECOND image is the NEWER capture.

Analyse what has changed between these two images and provide:

1. OVERALL CHANGE SUMMARY (2 sentences)
   What is the most significant change you observe?

2. VEGETATION / FOREST CHANGE
   Has forest or green coverage increased or decreased?
   Describe where and how much approximately.

3. URBAN DEVELOPMENT
   Is there new construction, roads, or built-up area visible?

4. WATER BODIES
   Have any rivers, lakes, or water areas changed in size?

5. DEFORESTATION EVIDENCE
   Do you see clear-cut areas, burning scars, or forest edges
   that appear to be retreating?

6. CONFIDENCE LEVEL
   How confident are you in your assessment? (Low/Medium/High)
   What makes it difficult or easy to assess?

Be specific about locations within the image (top-left, centre, etc.)
"""
            response = model.generate_content([prompt, old_part, new_part])
            logger.info("Two-image visual comparison generated via Gemini Vision.")
            return response.text

        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            return f"Visual comparison unavailable: {str(e)}"


# ══════════════════════════════════════════════════════════════
# 4. LOCATION INTELLIGENCE
# ══════════════════════════════════════════════════════════════

class LocationIntelligence:
    """
    Enriches analysis with location-specific context using Gemini.

    If user provides coordinates or region name, Gemini retrieves:
    - Known deforestation drivers in that region
    - Protected area status
    - Local environmental policies
    - Historical deforestation context
    - Nearby threatened species
    """

    def get_location_context(self,
                              location_description: str,
                              change_results: dict) -> str:
        """
        Generate location-specific environmental intelligence.

        Parameters:
            location_description (str): User-provided location
                e.g. "Amazon, Brazil" or "Congo Basin" or "15.2°N, 100.3°E"
            change_results (dict): Full change detection results.

        Returns:
            str: AI-generated location-specific context and analysis.
        """
        prompt = f"""
You are a regional environmental intelligence expert with deep knowledge
of global deforestation patterns, protected areas, and conservation policy.

LOCATION: {location_description}

SATELLITE ANALYSIS RESULTS FOR THIS LOCATION:
  Forest loss       : {change_results['forest_loss_pct']:.1f}%
  Urban expansion   : {change_results['urban_expansion_pct']:.1f}%
  Water change      : {change_results['water_change_pct']:+.1f}%
  Risk level        : {change_results['risk_level']}
  Total change      : {change_results['total_changed_pct']:.1f}%

Provide location-specific intelligence covering:

1. REGIONAL DEFORESTATION CONTEXT (2-3 sentences)
   Is this level of forest loss typical or unusual for this region?
   What are the known primary drivers of deforestation here?

2. PROTECTED AREA STATUS (1-2 sentences)
   Are there known protected areas, national parks, or
   indigenous territories in or near this region?

3. POLICY & GOVERNANCE (2 sentences)
   What environmental laws or international agreements apply here?
   Are they being enforced effectively?

4. THREATENED SPECIES (2-3 bullet points)
   Which specific species endemic to this region are most
   affected by this level of forest loss?

5. REGIONAL DEFORESTATION RATE COMPARISON (1 sentence)
   How does the detected {change_results['forest_loss_pct']:.1f}% loss
   compare to the known annual deforestation rate for this region?

Be specific to this location. Use real regional knowledge.
If coordinates are provided, identify the country/biome.
"""
        try:
            model    = _get_model()
            response = model.generate_content(prompt)
            logger.info(f"Location intelligence generated for: {location_description}")
            return response.text
        except Exception as e:
            return f"Location intelligence unavailable: {str(e)}"


# ══════════════════════════════════════════════════════════════
# 5. SPECIES IMPACT AI
# ══════════════════════════════════════════════════════════════

class SpeciesImpactAI:
    """
    AI-powered biodiversity and species impact estimator.
    Uses Gemini to reason about ecological consequences of
    detected forest loss beyond simple percentage numbers.
    """

    def estimate_species_impact(self,
                                 change_results: dict,
                                 region_type: str = "tropical forest") -> str:
        """
        Generate an AI assessment of which species and ecosystems
        are most threatened by the detected deforestation.

        Parameters:
            change_results (dict): Full change detection results.
            region_type (str): Biome/region description for context.

        Returns:
            str: AI-generated species impact assessment.
        """
        prompt = f"""
You are a conservation biologist and biodiversity expert.

DEFORESTATION ANALYSIS:
  Region type       : {region_type}
  Forest loss       : {change_results['forest_loss_pct']:.1f}% of observed area
  Forest gain       : {change_results['forest_gain_pct']:.1f}% (reforestation)
  Net forest change : {change_results['net_forest_change_pct']:+.1f}%
  Urban expansion   : {change_results['urban_expansion_pct']:.1f}%
  Risk level        : {change_results['risk_level']}

Provide a biodiversity impact assessment covering:

1. HABITAT FRAGMENTATION (2 sentences)
   How does this level of forest loss affect habitat connectivity?
   What is the minimum viable habitat concern here?

2. MOST VULNERABLE SPECIES (3-4 bullet points)
   List specific species categories or examples most at risk
   in {region_type} from this level of deforestation.
   Include mammals, birds, amphibians, and insects if relevant.

3. EDGE EFFECT ANALYSIS (1-2 sentences)
   Beyond the lost area, how much additional habitat is degraded
   by forest edge effects at this loss percentage?

4. ECOSYSTEM SERVICES LOST (2-3 bullet points)
   What ecosystem services (water regulation, carbon sequestration,
   pollination, soil stability) are compromised?

5. RECOVERY POTENTIAL (1-2 sentences)
   Given the {change_results['forest_gain_pct']:.1f}% reforestation gain,
   what is the realistic recovery timeline for biodiversity?

Be scientifically specific. Reference real ecological principles.
"""
        try:
            model    = _get_model()
            response = model.generate_content(prompt)
            logger.info("Species impact assessment generated.")
            return response.text
        except Exception as e:
            return f"Species impact assessment unavailable: {str(e)}"


# ══════════════════════════════════════════════════════════════
# 6. SMART ALERT SYSTEM
# ══════════════════════════════════════════════════════════════

class SmartAlertSystem:
    """
    AI-driven alert generator that decides WHAT to alert about,
    WHO should be alerted, and HOW urgently — based on the
    combination of signals rather than simple thresholds.

    Goes beyond: "risk > 25% → alert"
    Instead:     "forest loss + urban expansion + water loss in
                  the same area → likely illegal logging for
                  agricultural conversion → alert FAO + local authority"
    """

    def generate_smart_alerts(self, change_results: dict) -> dict:
        """
        Generate intelligent, contextual alerts based on the
        combination of detected environmental signals.

        Parameters:
            change_results (dict): Full change detection results.

        Returns:
            dict: {
                'alerts'      : list of alert dicts,
                'ai_narrative': str — Gemini's alert reasoning,
                'priority'    : str — Overall alert priority
            }
        """
        # Rule-based signal detection (these are facts, not AI)
        signals = []

        if change_results["forest_loss_pct"] > 5:
            signals.append(f"Forest loss: {change_results['forest_loss_pct']:.1f}%")

        if change_results["urban_expansion_pct"] > 3:
            signals.append(f"Urban expansion: {change_results['urban_expansion_pct']:.1f}%")

        if change_results["water_change_pct"] < -2:
            signals.append(f"Water body reduction: {change_results['water_change_pct']:.1f}%")

        if change_results["num_changed_regions"] > 5:
            signals.append(f"Multiple change zones: {change_results['num_changed_regions']} regions")

        # Composite alert priority
        score = (
            change_results["forest_loss_pct"] * 0.5 +
            change_results["urban_expansion_pct"] * 0.3 +
            abs(min(change_results["water_change_pct"], 0)) * 0.2
        )

        if score > 30:
            priority = "CRITICAL"
        elif score > 15:
            priority = "HIGH"
        elif score > 5:
            priority = "MEDIUM"
        else:
            priority = "LOW"

        # AI reasoning about what combination of signals means
        ai_narrative = self._generate_alert_reasoning(
            signals, change_results, priority
        )

        # Structured alerts
        alerts = self._build_alert_list(change_results, priority)

        return {
            "alerts"       : alerts,
            "ai_narrative" : ai_narrative,
            "priority"     : priority,
            "signals"      : signals
        }

    def _generate_alert_reasoning(self,
                                   signals: list,
                                   change_results: dict,
                                   priority: str) -> str:
        """Use Gemini to reason about what the combination of signals means."""
        if not signals:
            return "No significant environmental alerts detected in this analysis."

        prompt = f"""
You are an environmental monitoring system AI.

DETECTED SIGNALS:
{chr(10).join(f"  • {s}" for s in signals)}

FULL CONTEXT:
  Forest loss       : {change_results['forest_loss_pct']:.1f}%
  Urban expansion   : {change_results['urban_expansion_pct']:.1f}%
  Water change      : {change_results['water_change_pct']:+.1f}%
  Changed regions   : {change_results['num_changed_regions']}
  Risk level        : {change_results['risk_level']}
  Alert priority    : {priority}

In 2-3 sentences:
1. What does this COMBINATION of signals most likely indicate?
   (e.g. illegal logging, agricultural expansion, urban sprawl,
   climate-driven die-off, mining activity, etc.)
2. Who specifically should be alerted?
   (e.g. national forestry authority, FAO, local government,
   indigenous community leaders, NGOs)
3. What is the most time-sensitive action needed?

Be direct and specific. This is an operational alert.
"""
        try:
            model    = _get_model()
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            return (f"Alert priority: {priority}. "
                    f"Detected signals: {', '.join(signals)}. "
                    f"Immediate review recommended.")

    def _build_alert_list(self,
                           change_results: dict,
                           priority: str) -> list:
        """Build structured alert objects for dashboard display."""
        alerts = []
        ts     = datetime.now().strftime("%Y-%m-%d %H:%M")

        if change_results["forest_loss_pct"] > 5:
            alerts.append({
                "type"     : "DEFORESTATION",
                "priority" : priority,
                "message"  : (f"Forest loss of {change_results['forest_loss_pct']:.1f}% "
                              f"detected in observed area."),
                "timestamp": ts,
                "icon"     : "🌲"
            })

        if change_results["urban_expansion_pct"] > 3:
            alerts.append({
                "type"     : "URBAN ENCROACHMENT",
                "priority" : "HIGH" if change_results["urban_expansion_pct"] > 8 else "MEDIUM",
                "message"  : (f"Urban area expanded by "
                              f"{change_results['urban_expansion_pct']:.1f}%, "
                              f"potentially encroaching on natural habitat."),
                "timestamp": ts,
                "icon"     : "🏙️"
            })

        if change_results["water_change_pct"] < -2:
            alerts.append({
                "type"     : "WATER BODY REDUCTION",
                "priority" : "HIGH" if change_results["water_change_pct"] < -5 else "MEDIUM",
                "message"  : (f"Water coverage reduced by "
                              f"{abs(change_results['water_change_pct']):.1f}%, "
                              f"indicating possible hydrological stress."),
                "timestamp": ts,
                "icon"     : "💧"
            })

        if not alerts:
            alerts.append({
                "type"     : "STATUS",
                "priority" : "LOW",
                "message"  : "No significant environmental alerts. Area appears stable.",
                "timestamp": ts,
                "icon"     : "✅"
            })

        return alerts


# ══════════════════════════════════════════════════════════════
# 7. ANOMALY DETECTOR
# ══════════════════════════════════════════════════════════════

class AnomalyDetector:
    """
    Statistical anomaly detection for change patterns.
    Flags regions where change magnitude is statistically
    unusual compared to the image-wide distribution.
    """

    def detect_anomalies(self,
                          fused_change_map: np.ndarray,
                          contour_stats: pd.DataFrame) -> dict:
        """
        Identify statistically anomalous change regions using
        Z-score analysis on contour area distribution.

        Parameters:
            fused_change_map (np.ndarray): Grayscale change intensity map.
            contour_stats (pd.DataFrame): Per-region contour statistics.

        Returns:
            dict: Anomaly detection results with flagged regions.
        """
        if contour_stats.empty:
            return {"anomalies": [], "summary": "No contours to analyse."}

        # Statistical analysis of change intensity
        change_values   = fused_change_map.flatten().astype(np.float32)
        mean_change     = float(np.mean(change_values))
        std_change      = float(np.std(change_values))
        p95_threshold   = float(np.percentile(change_values, 95))

        # Flag contours with unusually large area (Z-score > 2)
        if len(contour_stats) > 1:
            area_mean = contour_stats["Area_px"].mean()
            area_std  = contour_stats["Area_px"].std()

            if area_std > 0:
                contour_stats = contour_stats.copy()
                contour_stats["Z_Score"] = (
                    (contour_stats["Area_px"] - area_mean) / area_std
                )
                anomalous = contour_stats[
                    contour_stats["Z_Score"] > 2.0
                ].to_dict("records")
            else:
                anomalous = []
        else:
            anomalous = []

        # Hotspot pixels — top 5% change intensity
        hotspot_pct = float(
            np.sum(fused_change_map > p95_threshold) / fused_change_map.size * 100
        )

        result = {
            "anomalies"          : anomalous,
            "num_anomalies"      : len(anomalous),
            "mean_change"        : round(mean_change, 2),
            "std_change"         : round(std_change, 2),
            "p95_threshold"      : round(p95_threshold, 2),
            "hotspot_area_pct"   : round(hotspot_pct, 2),
            "summary"            : (
                f"{len(anomalous)} statistically anomalous change regions detected. "
                f"Top 5% intensity hotspots cover {hotspot_pct:.1f}% of the image."
            )
        }

        logger.info(f"Anomaly detection: {len(anomalous)} anomalies | "
                    f"Hotspot area: {hotspot_pct:.1f}%")
        return result


# ══════════════════════════════════════════════════════════════
# STREAMLIT UI COMPONENTS
# ══════════════════════════════════════════════════════════════

def render_carbon_section(change_results: dict):
    """Render the Carbon Stock Estimator section in Streamlit."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown(
        '<div class="section-header">🌱 Carbon Stock & CO₂ Impact</div>',
        unsafe_allow_html=True
    )

    estimator = CarbonEstimator()

    col1, col2 = st.columns([1, 2])
    with col1:
        forest_type = st.selectbox(
            "Forest Type",
            ["tropical", "temperate", "boreal", "unknown"],
            help="Select the biome type for accurate carbon density estimation."
        )
        image_area = st.number_input(
            "Estimated Image Area (hectares)",
            min_value=100,
            max_value=1000000,
            value=10000,
            step=1000,
            help="Approximate real-world area covered by the satellite image."
        )

    carbon_data = estimator.estimate_carbon_loss(
        forest_loss_pct = change_results["forest_loss_pct"],
        image_area_ha   = image_area,
        forest_type     = forest_type
    )

    with col2:
        c1, c2, c3 = st.columns(3)
        c1.metric("Carbon Lost (tonnes)",  f"{carbon_data['carbon_tonnes']:,}")
        c2.metric("CO₂ Equivalent (t)",   f"{carbon_data['co2_tonnes']:,}")
        c3.metric("Lost Area (ha)",        f"{carbon_data['lost_area_ha']:,}")

    st.markdown(
        f'<div class="finding-block">'
        f'<strong>Real-World Equivalents:</strong><br>'
        f'🚗 {carbon_data["equivalent_cars"]:,} cars driven for one year<br>'
        f'✈️ {carbon_data["equivalent_flights"]:,} London–New York flights<br>'
        f'🏠 {carbon_data["equivalent_homes"]:,} homes powered for one year'
        f'</div>',
        unsafe_allow_html=True
    )

    if GEMINI_API_KEY:
        if st.button("🤖 Generate AI Carbon Analysis",
                     use_container_width=True):
            with st.spinner("Generating carbon impact analysis..."):
                narrative = estimator.generate_ai_carbon_report(
                    carbon_data, change_results
                )
                st.markdown(
                    f'<div class="summary-box">{narrative}</div>',
                    unsafe_allow_html=True
                )


def render_trend_section(change_results: dict):
    """Render the Forest Trend Predictor section in Streamlit."""
    try:
        import streamlit as st
        import plotly.graph_objects as go
    except ImportError:
        return

    st.markdown(
        '<div class="section-header">📈 Forest Cover Forecast</div>',
        unsafe_allow_html=True
    )

    predictor = TrendPredictor()

    col1, col2 = st.columns(2)
    with col1:
        years_between = st.slider(
            "Years between image captures",
            min_value=1, max_value=20, value=2,
            help="How many years apart were the two satellite images taken?"
        )
    with col2:
        forecast_years = st.slider(
            "Forecast horizon (years)",
            min_value=5, max_value=30, value=10
        )

    trend = predictor.predict_forest_trajectory(
        old_forest_pct = change_results["old_forest_pct"],
        new_forest_pct = change_results["new_forest_pct"],
        years_between  = years_between,
        forecast_years = forecast_years
    )

    df = predictor.build_forecast_chart_data(trend)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["Year"], y=df["Forest_Pct"],
        mode="lines+markers",
        name="Projected Forest Cover",
        line=dict(color="#E74C3C", width=2, dash="dash"),
        marker=dict(size=5)
    ))

    # Add current point
    fig.add_trace(go.Scatter(
        x=[df["Year"].iloc[0]],
        y=[trend["current_forest_pct"]],
        mode="markers",
        name="Current",
        marker=dict(color="#27AE60", size=12, symbol="star")
    ))

    # Critical threshold line
    fig.add_hline(
        y=10, line_dash="dot",
        line_color="#E74C3C",
        annotation_text="Critical threshold (10%)"
    )

    fig.update_layout(
        title="Forest Cover Projection",
        xaxis_title="Year",
        yaxis_title="Forest Cover (%)",
        yaxis=dict(range=[0, max(trend["current_forest_pct"] * 1.2, 20)]),
        plot_bgcolor="#F8F9FA",
        paper_bgcolor="white",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Annual Change",
                 f"{trend['annual_change_pct']:+.2f}%/yr")
    col_b.metric("Trend Direction",
                 trend["trend_direction"].title())
    col_c.metric("Years to Critical (<10%)",
                 str(trend.get("years_to_critical_10", "N/A")))

    if GEMINI_API_KEY:
        if st.button("🤖 Generate AI Forecast Narrative",
                     use_container_width=True,
                     key="forecast_btn"):
            with st.spinner("Generating forecast..."):
                narrative = predictor.generate_ai_forecast_narrative(
                    trend, change_results
                )
                st.markdown(
                    f'<div class="summary-box">{narrative}</div>',
                    unsafe_allow_html=True
                )


def render_vision_section(old_bytes: bytes, new_bytes: bytes):
    """Render Gemini Vision direct image analysis section."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown(
        '<div class="section-header">👁️ Gemini Vision Direct Analysis</div>',
        unsafe_allow_html=True
    )

    if not GEMINI_API_KEY:
        st.info("Gemini API key required for Vision analysis.")
        return

    captioner = ImageCaptioner()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔍 Analyse Old Image with Vision",
                     use_container_width=True):
            with st.spinner("Gemini Vision analysing old image..."):
                caption = captioner.caption_single_image(old_bytes)
                st.session_state["vision_old"] = caption

    with col2:
        if st.button("🔍 Analyse New Image with Vision",
                     use_container_width=True):
            with st.spinner("Gemini Vision analysing new image..."):
                caption = captioner.caption_single_image(new_bytes)
                st.session_state["vision_new"] = caption

    if st.button("🔄 Compare Both Images with Vision",
                 use_container_width=True, type="primary"):
        with st.spinner("Gemini Vision comparing both images..."):
            comparison = captioner.compare_two_images(old_bytes, new_bytes)
            st.session_state["vision_comparison"] = comparison

    # Display results
    for key, label in [
        ("vision_old",        "Old Image Analysis"),
        ("vision_new",        "New Image Analysis"),
        ("vision_comparison", "Visual Change Comparison")
    ]:
        if key in st.session_state and st.session_state[key]:
            st.markdown(f"**{label}**")
            st.markdown(
                f'<div class="finding-block">'
                f'{st.session_state[key]}'
                f'</div>',
                unsafe_allow_html=True
            )


def render_location_section(change_results: dict):
    """Render Location Intelligence section."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown(
        '<div class="section-header">🗺️ Location Intelligence</div>',
        unsafe_allow_html=True
    )

    if not GEMINI_API_KEY:
        st.info("Gemini API key required for location intelligence.")
        return

    location = st.text_input(
        "Enter location (name or coordinates)",
        placeholder="e.g. Amazon, Brazil  |  Congo Basin  |  15.2°N, 100.3°E",
        help="Provide the geographic location of your satellite images."
    )

    if st.button("🌍 Get Location Intelligence",
                 use_container_width=True,
                 disabled=not location):
        with st.spinner(f"Fetching intelligence for {location}..."):
            intel = LocationIntelligence()
            result = intel.get_location_context(location, change_results)
            st.session_state["location_intel"] = result

    if "location_intel" in st.session_state and st.session_state["location_intel"]:
        st.markdown(
            f'<div class="summary-box">'
            f'{st.session_state["location_intel"]}'
            f'</div>',
            unsafe_allow_html=True
        )


def render_species_section(change_results: dict):
    """Render Species Impact AI section."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown(
        '<div class="section-header">🦋 Biodiversity & Species Impact</div>',
        unsafe_allow_html=True
    )

    if not GEMINI_API_KEY:
        st.info("Gemini API key required for species impact analysis.")
        return

    region_type = st.selectbox(
        "Region / Biome Type",
        ["tropical forest", "temperate forest", "boreal forest",
         "savanna / grassland", "wetland", "montane forest",
         "coastal / mangrove", "unknown"]
    )

    if st.button("🦁 Generate Species Impact Report",
                 use_container_width=True):
        with st.spinner("Analysing biodiversity impact..."):
            species_ai = SpeciesImpactAI()
            result = species_ai.estimate_species_impact(
                change_results, region_type
            )
            st.session_state["species_impact"] = result

    if "species_impact" in st.session_state and st.session_state["species_impact"]:
        st.markdown(
            f'<div class="finding-block">'
            f'{st.session_state["species_impact"]}'
            f'</div>',
            unsafe_allow_html=True
        )


def render_alerts_section(change_results: dict):
    """Render Smart Alert System section."""
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown(
        '<div class="section-header">🚨 Smart Alert System</div>',
        unsafe_allow_html=True
    )

    alert_system = SmartAlertSystem()
    alert_data   = alert_system.generate_smart_alerts(change_results)

    priority_colors = {
        "CRITICAL": "#8E0000",
        "HIGH"    : "#E74C3C",
        "MEDIUM"  : "#F39C12",
        "LOW"     : "#27AE60"
    }
    p_color = priority_colors.get(alert_data["priority"], "#6C757D")

    st.markdown(
        f'<div style="background:{p_color}18;border:2px solid {p_color}55;'
        f'border-radius:10px;padding:14px 20px;margin-bottom:16px;">'
        f'<strong style="color:{p_color}">Overall Alert Priority: '
        f'{alert_data["priority"]}</strong>'
        f'</div>',
        unsafe_allow_html=True
    )

    for alert in alert_data["alerts"]:
        ac = priority_colors.get(alert["priority"], "#6C757D")
        st.markdown(
            f'<div class="finding-block">'
            f'{alert["icon"]} <strong>[{alert["type"]}]</strong> '
            f'<span style="color:{ac};font-weight:600">{alert["priority"]}</span><br>'
            f'{alert["message"]}<br>'
            f'<span style="font-size:0.8em;color:#6C757D">{alert["timestamp"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    if alert_data["ai_narrative"] and GEMINI_API_KEY:
        st.markdown("**🤖 AI Alert Reasoning**")
        st.markdown(
            f'<div class="summary-box">{alert_data["ai_narrative"]}</div>',
            unsafe_allow_html=True
        )