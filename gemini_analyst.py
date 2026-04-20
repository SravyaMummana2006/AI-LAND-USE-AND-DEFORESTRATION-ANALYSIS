"""
gemini_analyst.py
-----------------
Gemini API Integration for AI-Powered Analysis & Chatbot.

Responsibilities:
- Generate genuinely intelligent natural-language reports using Gemini
- Power an interactive Q&A chatbot about the satellite analysis
- Provide contextual environmental insights beyond template sentences
- Suggest location-specific conservation recommendations
- Answer follow-up user questions about deforestation causes and impacts

Requirements:
    pip install google-generativeai

Setup:
    1. Get free API key: https://aistudio.google.com/app/apikey
    2. Set environment variable:
       export GEMINI_API_KEY="your_key_here"       (Linux/macOS)
       set GEMINI_API_KEY=your_key_here            (Windows)
    OR create a .env file:
       GEMINI_API_KEY=your_key_here

Author: AI Land Use Analysis System
"""

import os
import logging
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Load API Key
# ─────────────────────────────────────────────
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = "models/gemini-2.5-flash"   # Fast, free tier available


def initialise_gemini(api_key: str = GEMINI_API_KEY) -> bool:
    """
    Initialise the Gemini API client with the provided API key.

    Parameters:
        api_key (str): Gemini API key from Google AI Studio.

    Returns:
        bool: True if initialised successfully, False otherwise.
    """
    if not api_key:
        logger.warning(
            "GEMINI_API_KEY not set. "
            "Get a free key at: https://aistudio.google.com/app/apikey"
        )
        return False

    try:
        genai.configure(api_key=api_key)
        logger.info("Gemini API initialised successfully.")
        return True
    except Exception as e:
        logger.error(f"Gemini init failed: {e}")
        return False


# ─────────────────────────────────────────────
# Context Builder
# ─────────────────────────────────────────────

def build_analysis_context(change_results: dict,
                            old_area_stats,
                            new_area_stats,
                            prediction_comparison: dict) -> str:
    """
    Build a structured text context block from all analysis results.
    This is injected into every Gemini prompt so it has full knowledge
    of the satellite analysis findings.

    Parameters:
        change_results (dict): Output from detect_change.run_change_detection().
        old_area_stats: DataFrame from predict.run_full_prediction().
        new_area_stats: DataFrame from predict.run_full_prediction().
        prediction_comparison (dict): Output from predict.compare_predictions().

    Returns:
        str: Structured context paragraph for Gemini prompts.
    """
    # Format area stats as readable text
    old_stats_text = "\n".join([
        f"    {row['Class']}: {row['Percentage']:.1f}%"
        for _, row in old_area_stats.iterrows()
    ])
    new_stats_text = "\n".join([
        f"    {row['Class']}: {row['Percentage']:.1f}%"
        for _, row in new_area_stats.iterrows()
    ])

    # Format class-level changes
    changes_text = "\n".join([
        f"    {cls}: {vals['old_pct']:.1f}% → {vals['new_pct']:.1f}% "
        f"({vals['direction']}, {abs(vals['change_pct']):.1f}% change)"
        for cls, vals in prediction_comparison.items()
    ])

    context = f"""
SATELLITE IMAGE ANALYSIS RESULTS
==================================
FOREST METRICS:
  Old forest cover        : {change_results['old_forest_pct']:.1f}%
  New forest cover        : {change_results['new_forest_pct']:.1f}%
  Forest loss             : {change_results['forest_loss_pct']:.1f}%
  Forest gain (reforest.) : {change_results['forest_gain_pct']:.1f}%
  Net forest change       : {change_results['net_forest_change_pct']:+.1f}%

URBAN METRICS:
  Old urban coverage      : {change_results['old_urban_pct']:.1f}%
  New urban coverage      : {change_results['new_urban_pct']:.1f}%
  Urban expansion         : {change_results['urban_expansion_pct']:.1f}%

WATER METRICS:
  Old water coverage      : {change_results['old_water_pct']:.1f}%
  New water coverage      : {change_results['new_water_pct']:.1f}%
  Water body change       : {change_results['water_change_pct']:+.1f}%

OVERALL CHANGE:
  Total changed area      : {change_results['total_changed_pct']:.1f}%
  Number of changed zones : {change_results['num_changed_regions']}
  Risk classification     : {change_results['risk_level']}

LAND USE COVERAGE — OLD IMAGE:
{old_stats_text}

LAND USE COVERAGE — NEW IMAGE:
{new_stats_text}

PER-CLASS CHANGES:
{changes_text}
==================================
"""
    return context


# ─────────────────────────────────────────────
# Report Generation
# ─────────────────────────────────────────────

def generate_ai_report(analysis_context: str) -> str:
    """
    Use Gemini to generate a genuinely intelligent, contextual
    environmental analysis report — not a template.

    Gemini reasons about the combination of signals (forest loss +
    urban expansion + water change) to produce expert-level insights
    that template systems cannot match.

    Parameters:
        analysis_context (str): Structured analysis data string.

    Returns:
        str: Full AI-generated report text.
    """
    prompt = f"""
You are an expert environmental scientist and remote sensing analyst.
You have been provided with the results of an AI satellite image analysis
comparing two time-period images of the same geographic area.

{analysis_context}

Based on these results, write a comprehensive, intelligent environmental
analysis report. Your report must:

1. EXECUTIVE SUMMARY (2-3 sentences)
   Describe the overall environmental situation and its severity.

2. FOREST ANALYSIS (3-4 sentences)
   Analyse the forest change in depth. What might be causing it?
   What are the ecological consequences of this level of loss?

3. URBAN & LAND USE DYNAMICS (2-3 sentences)
   Describe the relationship between urban expansion and other land changes.
   Is there evidence of urban encroachment on natural areas?

4. WATER & HYDROLOGICAL IMPACT (2 sentences)
   Analyse water body changes and their environmental implications.

5. RISK ASSESSMENT (2-3 sentences)
   Justify the risk level. What makes this situation serious or manageable?
   What are the long-term consequences if trends continue?

6. SPECIFIC RECOMMENDATIONS (3-4 bullet points)
   Give concrete, actionable conservation recommendations.
   Be specific — not generic advice.

7. CARBON & BIODIVERSITY IMPACT (2 sentences)
   Estimate the likely carbon stock impact and biodiversity consequences.

Write in a professional but accessible tone. Be specific about numbers
from the data. Do NOT use generic filler phrases. Every sentence must
add real analytical value.
"""

    try:
        model    = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        if hasattr(response, "text"):
         report = response.text
        else:
         report = "No response generated."

        logger.info("Gemini AI report generated successfully.")
        return report

    except Exception as e:
        logger.error(f"Gemini report generation failed: {e}")
        return f"AI report generation unavailable: {str(e)}"


# ─────────────────────────────────────────────
# Chatbot
# ─────────────────────────────────────────────

class DeforestationChatbot:
    """
    Interactive chatbot that answers user questions about the
    satellite analysis using Gemini with full conversation memory.

    The chatbot is pre-loaded with the analysis context so it can
    answer specific questions like:
        "Why is forest cover decreasing in this area?"
        "What will happen if this trend continues for 10 years?"
        "What policies could help reverse deforestation here?"
        "Is the water body change related to the forest loss?"
    """

    def __init__(self, analysis_context: str):
        """
        Initialise the chatbot with the satellite analysis context.

        Parameters:
            analysis_context (str): Full analysis results context string.
        """
        self.analysis_context = analysis_context
        self.conversation_history = []

        # System instruction injected at the start of every conversation
        self.system_prompt = f"""
You are an expert AI environmental analyst and deforestation specialist.
You have access to the results of a satellite image analysis comparing
two time periods of the same geographic area.

Here are the complete analysis results you must use to answer questions:

{analysis_context}

Guidelines for your responses:
- Always refer to specific numbers from the analysis data
- Connect different signals (forest loss + urban expansion + water change)
- Explain causes and consequences, not just describe numbers
- Be concise but insightful — avoid vague generalisations
- If asked about something outside the analysis scope, say so clearly
- Use a professional but accessible tone
- When relevant, mention conservation strategies, policies, or interventions
"""

    def chat(self, user_message: str) -> str:
        """
        Send a user message and get an AI response.
        Maintains full conversation history for contextual follow-ups.

        Parameters:
            user_message (str): The user's question or message.

        Returns:
            str: Gemini's response.
        """
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)

            # Build conversation with system context + history
            messages = []

            # Inject system context as first user turn
            if not self.conversation_history:
                messages.append({
                    "role"  : "user",
                    "parts" : [self.system_prompt +
                               "\n\nAcknowledge that you have received the "
                               "analysis data and are ready to answer questions."]
                })
                messages.append({
                    "role"  : "model",
                    "parts" : ["I have received and analysed the satellite "
                               "imagery results. I can see the forest coverage, "
                               "urban expansion, water changes, and risk "
                               "assessment data. I'm ready to answer your "
                               "questions about this environmental analysis."]
                })

            # Add conversation history
            messages.extend(self.conversation_history)

            # Add current user message
            messages.append({
                "role"  : "user",
                "parts" : [user_message]
            })

            # Generate response
            chat_session = model.start_chat(history=messages[:-1])
            response     = chat_session.send_message(user_message)
            if hasattr(response, "text"):
             ai_reply = response.text
            else:
             ai_reply = "No response generated."

            # Update history
            self.conversation_history.append({
                "role" : "user", "parts": [user_message]
            })
            self.conversation_history.append({
                "role" : "model", "parts": [ai_reply]
            })

            logger.info(f"Chatbot response generated | "
                        f"History length: {len(self.conversation_history)}")
            return ai_reply

        except Exception as e:
            error_msg = f"Chatbot error: {str(e)}"
            logger.error(error_msg)
            return (
                f"I encountered an error processing your question. "
                f"Please check your Gemini API key. Error: {str(e)}"
            )

    def reset_conversation(self):
        """Clear conversation history to start a fresh chat session."""
        self.conversation_history = []
        logger.info("Chatbot conversation history reset.")

    def get_suggested_questions(self) -> list:
        """
        Return context-aware suggested questions based on the
        analysis results — shown as quick-tap buttons in the dashboard.

        Returns:
            list: List of suggested question strings.
        """
        base_questions = [
            "What is likely causing the forest loss in this area?",
            "What will happen if this deforestation trend continues for 10 years?",
            "Is the urban expansion directly responsible for the forest loss?",
            "What conservation policies would be most effective here?",
            "How does the water body change relate to the deforestation?",
            "What is the estimated carbon stock lost from this deforestation?",
            "Which biodiversity species are most at risk in this area?",
            "How does this compare to global deforestation rates?",
        ]
        return base_questions


# ─────────────────────────────────────────────
# Streamlit UI Components
# ─────────────────────────────────────────────

def render_gemini_report_section(analysis_context: str):
    """
    Render the Gemini AI report section in the Streamlit dashboard.
    Shows a 'Generate AI Report' button and displays the result.

    Parameters:
        analysis_context (str): Full analysis context string.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown(
        '<div class="section-header">🤖 Gemini AI Report</div>',
        unsafe_allow_html=True
    )

    if not GEMINI_API_KEY:
        st.warning(
            "⚠️ Gemini API key not set. "
            "Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey) "
            "and add it to your `.env` file as `GEMINI_API_KEY=your_key`"
        )
        return

    if st.button("✨ Generate AI-Powered Report",
                 use_container_width=True,
                 type="primary"):
        with st.spinner("Gemini is analysing your satellite data..."):
            report = generate_ai_report(analysis_context)
            st.session_state["gemini_report"] = report

    if "gemini_report" in st.session_state and st.session_state["gemini_report"]:
        st.markdown(
            f'<div style="background:#F0F7FF;border-left:5px solid #2E86C1;'
            f'border-radius:0 10px 10px 0;padding:20px 24px;'
            f'font-size:0.96em;line-height:1.8;white-space:pre-wrap;">'
            f'{st.session_state["gemini_report"]}'
            f'</div>',
            unsafe_allow_html=True
        )

        # Download button for AI report
        st.download_button(
            label     = "⬇️ Download AI Report",
            data      = st.session_state["gemini_report"],
            file_name = "gemini_ai_report.txt",
            mime      = "text/plain"
        )


def render_chatbot_section(chatbot: "DeforestationChatbot"):
    """
    Render the interactive Gemini chatbot section in the Streamlit dashboard.
    Includes suggested question buttons, chat history, and input box.

    Parameters:
        chatbot (DeforestationChatbot): Initialised chatbot instance.
    """
    try:
        import streamlit as st
    except ImportError:
        return

    st.markdown(
        '<div class="section-header">💬 Ask the AI Analyst</div>',
        unsafe_allow_html=True
    )

    if not GEMINI_API_KEY:
        st.warning(
            "⚠️ Gemini API key required for the chatbot. "
            "Add `GEMINI_API_KEY=your_key` to your `.env` file."
        )
        return

    # Initialise chat history in session state
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Suggested questions as quick buttons
    st.markdown("**💡 Suggested Questions**")
    suggestions = chatbot.get_suggested_questions()

    cols = st.columns(2)
    for i, question in enumerate(suggestions[:4]):
        with cols[i % 2]:
            if st.button(
                f"❓ {question[:55]}{'...' if len(question) > 55 else ''}",
                key=f"suggest_{i}",
                use_container_width=True
            ):
                st.session_state.chat_messages.append({
                    "role": "user", "content": question
                })
                with st.spinner("Thinking..."):
                    reply = chatbot.chat(question)
                st.session_state.chat_messages.append({
                    "role": "assistant", "content": reply
                })

    st.markdown("---")

    # Chat history display
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_messages:
            if msg["role"] == "user":
                st.markdown(
                    f'<div style="background:#E8F4FD;border-radius:12px 12px 2px 12px;'
                    f'padding:12px 16px;margin:8px 0;margin-left:15%;'
                    f'font-size:0.95em;">'
                    f'<strong>You:</strong> {msg["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="background:#F0FAF0;border-radius:12px 12px 12px 2px;'
                    f'padding:12px 16px;margin:8px 0;margin-right:15%;'
                    f'font-size:0.95em;border-left:3px solid #52B788;">'
                    f'<strong>🤖 AI Analyst:</strong><br>{msg["content"]}</div>',
                    unsafe_allow_html=True
                )

    # User input
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    col_input, col_send, col_clear = st.columns([5, 1, 1])

    with col_input:
        user_input = st.text_input(
            "Ask anything about this analysis...",
            placeholder="e.g. What is causing the forest loss here?",
            label_visibility="collapsed",
            key="chat_input"
        )

    with col_send:
        send = st.button("Send", type="primary", use_container_width=True)

    with col_clear:
        if st.button("Clear", use_container_width=True):
            st.session_state.chat_messages = []
            chatbot.reset_conversation()
            st.rerun()

    # Process user input on send
    if send and user_input.strip():
        st.session_state.chat_messages.append({
            "role": "user", "content": user_input
        })
        with st.spinner("AI Analyst is thinking..."):
            reply = chatbot.chat(user_input)
        st.session_state.chat_messages.append({
            "role": "assistant", "content": reply
        })
        st.rerun()