"""
Streamlit frontend for LogiSense — Complaint Resolution Dashboard.

Provides an interactive UI for submitting customer complaints,
viewing resolution results, and inspecting the agent trace.

Usage:
    streamlit run frontend/app.py
"""

import json
import os
import sys
from pathlib import Path

import httpx
import streamlit as st

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ─────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────

try:
    API_URL = st.secrets.get("API_URL", os.getenv("API_URL", "http://localhost:8000"))
except Exception:
    API_URL = os.getenv("API_URL", "http://localhost:8000")

API_BASE_URL = API_URL

# Page config
st.set_page_config(
    page_title="LogiSense — AI Logistics Copilot",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main styling */
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .sub-header {
        text-align: center;
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Resolution badges */
    .badge-refund {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .badge-reschedule {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .badge-escalate {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
        padding: 0.5rem 1.5rem;
        border-radius: 25px;
        font-weight: 700;
        font-size: 1.1rem;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Severity badges */
    .severity-low {
        background-color: #d1fae5;
        color: #065f46;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .severity-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    .severity-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.9rem;
    }

    /* Trace step */
    .trace-step {
        padding: 0.85rem 1.15rem;
        margin: 0.6rem 0;
        border-left: 4px solid #6366f1;
        background-color: #ffffff;
        color: #0f172a !important;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
        border-radius: 0 10px 10px 0;
        font-size: 0.95rem;
        line-height: 1.5;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }

    .trace-step strong {
        color: #0f172a !important;
        font-weight: 700;
    }

    .trace-arrow {
        color: #6366f1 !important;
        font-weight: 700;
        margin: 0 0.35rem;
    }

    .trace-result {
        color: #1e1b4b !important;
        font-weight: 600;
    }

    /* Card styling */
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────

def call_api(endpoint: str, method: str = "GET", data: dict = None) -> dict:
    """Make an API call to the backend."""
    base_url = st.session_state.get("api_url", API_URL).rstrip("/")
    url = f"{base_url}{endpoint}"
    try:
        with httpx.Client(timeout=60.0) as client:
            if method == "POST":
                response = client.post(url, json=data)
            else:
                response = client.get(url)

            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code} — {response.text}"}
    except httpx.ConnectError:
        return {"error": f"Cannot connect to backend at {base_url}. Ensure the server is running."}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


def get_badge_html(action: str) -> str:
    """Get the HTML badge for a resolution action."""
    badge_class = f"badge-{action}" if action in ("refund", "reschedule", "escalate") else "badge-escalate"
    icons = {"refund": "💰", "reschedule": "📦", "escalate": "🚨"}
    icon = icons.get(action, "⚡")
    return f'<span class="{badge_class}">{icon} {action}</span>'


def get_severity_badge(severity: str) -> str:
    """Get the HTML badge for severity level."""
    sev_class = f"severity-{severity}" if severity in ("low", "medium", "high") else "severity-medium"
    return f'<span class="{sev_class}">{severity.upper()}</span>'


# ─────────────────────────────────────────────────
# Main App
# ─────────────────────────────────────────────────

def main():
    """Main Streamlit application."""

    # Header
    st.markdown('<h1 class="main-header">🚚 LogiSense</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Autonomous E-commerce Logistics Copilot — '
        'AI-powered complaint resolution with RAG, Fine-tuned BERT & Multi-Agent orchestration</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ──
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        api_url = st.text_input("API URL", value=API_URL, key="api_url")

        st.markdown("---")
        st.markdown("### 📊 System Status")
        if st.button("🔍 Check Health", key="health_btn"):
            health = call_api("/health")
            if "error" not in health:
                st.success(f"API: {health.get('status', 'unknown')}")
                components = health.get("components", {})
                for comp, status in components.items():
                    icon = "✅" if status == "healthy" else "⚠️" if status in ("not_initialized", "not_trained") else "❌"
                    st.markdown(f"{icon} **{comp}**: {status}")
            else:
                st.error(health["error"])

        st.markdown("---")
        st.markdown("### 📋 Quick Order Lookup")
        lookup_id = st.text_input("Order ID", placeholder="ORD-00001", key="lookup_id")
        if st.button("🔎 Look Up", key="lookup_btn"):
            if lookup_id:
                order = call_api(f"/order/{lookup_id}")
                if order.get("found"):
                    st.json(order)
                elif "error" in order:
                    st.error(order["error"])
                else:
                    st.warning(f"Order {lookup_id} not found.")

    # ── Main Content ──
    st.markdown("---")

    # Example complaint definitions
    examples = {
        "Low Severity": {
            "order_id": "ORD-00010",
            "text": "My order arrived a day late but everything looks fine. Just wanted to let you know.",
        },
        "Medium Severity": {
            "order_id": "ORD-00050",
            "text": "Order is delayed by 3 days now. I need it for an event this weekend. Please expedite.",
        },
        "High Severity": {
            "order_id": "ORD-00100",
            "text": "URGENT: Order never arrived and I was charged twice! I need immediate refund for both charges. This is unacceptable!",
        },
    }

    def _fill_example(label: str) -> None:
        """Callback to fill example data into session state."""
        st.session_state["order_id"] = examples[label]["order_id"]
        st.session_state["complaint_text"] = examples[label]["text"]

    # Input Form
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### 📝 Submit Complaint")
        customer_name = st.text_input(
            "Customer Name",
            placeholder="e.g., Aarav Sharma",
            key="customer_name",
        )
        order_id = st.text_input(
            "Order ID",
            placeholder="e.g., ORD-00001",
            key="order_id",
        )
        complaint_text = st.text_area(
            "Complaint Description",
            placeholder="Describe the issue with your order...",
            height=150,
            key="complaint_text",
        )

    with col2:
        st.markdown("### 💡 Example Complaints")

        for label in examples:
            st.button(
                f"📌 {label}",
                key=f"example_{label}",
                on_click=_fill_example,
                args=(label,),
            )


    # Submit button
    st.markdown("")
    submit_disabled = not (order_id and complaint_text and len(complaint_text) >= 10)

    if st.button("🚀 Resolve Complaint", disabled=submit_disabled, key="submit_btn"):
        with st.spinner("🔄 Running AI resolution pipeline..."):
            result = call_api(
                "/resolve",
                method="POST",
                data={
                    "order_id": order_id,
                    "complaint_text": complaint_text,
                },
            )

        if "error" in result and not result.get("resolution_action"):
            st.error(f"❌ {result['error']}")
        else:
            st.session_state["result"] = result

    # ── Results Display ──
    if "result" in st.session_state:
        result = st.session_state["result"]
        st.markdown("---")
        st.markdown("## 📊 Resolution Results")

        # Resolution summary cards
        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            action = result.get("resolution_action", "unknown")
            st.markdown(f"**Resolution Action**")
            st.markdown(get_badge_html(action), unsafe_allow_html=True)

        with res_col2:
            severity = result.get("severity", "unknown")
            confidence = result.get("severity_confidence", 0) * 100
            st.markdown(f"**Complaint Severity**")
            st.markdown(get_severity_badge(severity), unsafe_allow_html=True)
            st.caption(f"Confidence: {confidence:.1f}%")

        with res_col3:
            if result.get("refund_result"):
                rf = result["refund_result"]
                st.markdown("**Refund Status**")
                st.markdown(f"💰 **{rf.get('status', 'N/A').upper()}**")
                st.caption(f"Amount: ${rf.get('amount', 0):.2f}")
                st.caption(f"TX: {rf.get('transaction_id', 'N/A')}")
            else:
                st.markdown("**Refund Status**")
                st.markdown("N/A — No refund issued")

        # Reasoning
        st.markdown("### 💭 Resolution Reasoning")
        st.info(result.get("resolution_reasoning", "No reasoning provided."))

        # Email draft
        if result.get("email_draft"):
            email = result["email_draft"]
            st.markdown("### 📧 Email Draft")
            with st.expander(f"📨 {email.get('subject', 'Email Draft')}", expanded=False):
                st.markdown(f"**To:** {email.get('to', 'N/A')}")
                st.markdown(f"**Subject:** {email.get('subject', 'N/A')}")
                st.markdown(f"**Reference:** {email.get('reference_id', 'N/A')}")
                st.markdown("---")
                st.text(email.get("email_body", "No email body generated."))

        # Agent trace
        if result.get("agent_trace"):
            st.markdown("### 🔍 Agent Execution Trace")
            with st.expander("View step-by-step agent trace", expanded=True):
                for i, step in enumerate(result["agent_trace"], 1):
                    if isinstance(step, dict):
                        agent_name = step.get("agent_name", f"Agent {i}")
                        icon = step.get("icon", "🔹")
                        action = step.get("action", "Executed step")
                        result_text = step.get("result", "")
                    else:
                        # Fallback parsing if step is a raw string
                        step_str = str(step)
                        if "TrackerAgent" in step_str or "Tracker" in step_str:
                            icon = "🎯"
                            agent_name = "Tracker Agent"
                            action = "Fetched order status from database"
                            result_text = step_str.split("—")[-1].strip() if "—" in step_str else step_str
                        elif "RAGAgent" in step_str or "RAG" in step_str:
                            icon = "📊"
                            agent_name = "RAG Agent"
                            action = "Searched refund policy"
                            result_text = "Found: eligible for refund" if "refund" in step_str.lower() else "Retrieved policy guidelines"
                        elif "SentimentAgent" in step_str or "Sentiment" in step_str:
                            icon = "💬"
                            agent_name = "Sentiment Agent"
                            action = "Classified severity"
                            result_text = step_str.split("as")[-1].strip() if "as" in step_str else step_str
                        elif "ResolverAgent" in step_str or "Resolver" in step_str:
                            icon = "⚖️"
                            agent_name = "Resolver Agent"
                            action = "Made decision"
                            result_text = step_str.split("—")[-1].strip() if "—" in step_str else step_str
                        elif "ErrorHandler" in step_str:
                            icon = "❌"
                            agent_name = "Error Handler"
                            action = "Handled error"
                            result_text = step_str
                        else:
                            icon = "🔹"
                            agent_name = f"Agent Step {i}"
                            action = "Processed"
                            result_text = step_str

                    st.markdown(
                        f'<div class="trace-step">'
                        f'{icon} <strong>{agent_name}</strong> — {action} '
                        f'<span class="trace-arrow">→</span> '
                        f'<span class="trace-result">{result_text}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

        # Error display
        if result.get("error"):
            st.error(f"⚠️ Pipeline Error: {result['error']}")

    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #9ca3af; font-size: 0.85rem;">'
        '🚚 LogiSense v1.0 — Built with LangGraph, ChromaDB, BERT & FastAPI'
        '</p>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
