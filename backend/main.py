"""
FastAPI backend for the LogiSense Complaint Resolution System.

Endpoints:
    POST /resolve — Runs the full complaint resolution pipeline
    GET  /order/{order_id} — Returns order status
    GET  /health — API health check

Usage:
    uvicorn backend.main:app --reload --port 8000
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────────

class ComplaintRequest(BaseModel):
    """Request model for complaint resolution."""
    order_id: str = Field(..., description="Order ID to resolve", examples=["ORD-00001"])
    complaint_text: str = Field(
        ...,
        description="Customer complaint text",
        min_length=10,
        examples=["My order is delayed by 5 days and I need a refund."],
    )


class AgentTraceStep(BaseModel):
    """Structured step in the multi-agent execution trace."""
    step_number: int = 1
    step: Optional[int] = 1
    agent_name: str
    icon: str = "🔹"
    action: str
    result: str
    timestamp: str = ""
    formatted: Optional[str] = None


class ResolutionResponse(BaseModel):
    """Response model for complaint resolution."""
    order_id: str
    resolution_action: str
    resolution_reasoning: str
    severity: str
    severity_confidence: float = 0.0
    refund_result: Optional[dict] = None
    email_draft: Optional[dict] = None
    agent_trace: list[Union[AgentTraceStep, dict[str, Any], str]] = []
    error: Optional[str] = None


class OrderStatusResponse(BaseModel):
    """Response model for order status."""
    order_id: str
    customer_name: str = ""
    product: str = ""
    order_date: str = ""
    status: str = ""
    carrier: str = ""
    delivery_date: str = ""
    delay_reason: str = ""
    found: bool = False
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    version: str
    components: dict


# ─────────────────────────────────────────────────
# App Lifecycle
# ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.
    Starts immediately and binds to port without heavy ML loading.
    """
    logger.info("LogiSense API starting up (Lightweight Cloud Mode)...")
    logger.info("Server port binding immediately.")
    yield
    logger.info("LogiSense API shutting down...")


# ─────────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────────

app = FastAPI(
    title="LogiSense — Autonomous E-commerce Logistics Copilot",
    description=(
        "AI-powered complaint resolution system combining RAG, "
        "Sentiment Analysis, and Multi-Agent orchestration."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────

@app.post("/resolve", response_model=ResolutionResponse)
async def resolve_complaint(request: ComplaintRequest) -> ResolutionResponse:
    """
    Resolve a customer complaint using the multi-agent pipeline.

    Runs the full orchestration: tracker → rag → sentiment → resolver

    Args:
        request: ComplaintRequest with order_id and complaint_text.

    Returns:
        ResolutionResponse with the resolution decision and details.
    """
    logger.info(f"Resolve request: order_id={request.order_id}")

    try:
        from agents.orchestrator import resolve_complaint as run_pipeline

        result = run_pipeline(
            order_id=request.order_id,
            complaint_text=request.complaint_text,
        )

        return ResolutionResponse(
            order_id=request.order_id,
            resolution_action=result.get("resolution_action", "escalate"),
            resolution_reasoning=result.get("resolution_reasoning", ""),
            severity=result.get("severity", "unknown"),
            severity_confidence=result.get("severity_confidence", 0.0),
            refund_result=result.get("refund_result"),
            email_draft=result.get("email_draft"),
            agent_trace=result.get("agent_trace", []),
            error=result.get("error"),
        )

    except Exception as e:
        logger.error(f"Resolution error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Resolution pipeline error: {str(e)}",
        )


@app.get("/order/{order_id}", response_model=OrderStatusResponse)
async def get_order_status(order_id: str) -> OrderStatusResponse:
    """
    Get the status of an order.

    Args:
        order_id: The order ID to look up.

    Returns:
        OrderStatusResponse with order details.
    """
    logger.info(f"Order status request: {order_id}")

    try:
        from tools.order_db import get_order_status as lookup_order

        result = lookup_order(order_id)
        return OrderStatusResponse(**result)

    except Exception as e:
        logger.error(f"Order lookup error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Order lookup error: {str(e)}",
        )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """
    API health check endpoint.

    Returns status of the API and its component dependencies.
    """
    components = {
        "api": "healthy",
        "orders_db": "unknown",
        "vector_store": "unknown",
        "bert_model": "unknown",
    }

    # Check orders database
    try:
        orders_path = PROJECT_ROOT / "data" / "orders.csv"
        components["orders_db"] = "healthy" if orders_path.exists() else "missing"
    except Exception:
        components["orders_db"] = "error"

    # Check policy retriever availability
    try:
        policies_dir = PROJECT_ROOT / "data" / "policies"
        has_policies = policies_dir.exists() and any(policies_dir.glob("*.pdf"))
        components["vector_store"] = "healthy" if has_policies else "fallback_policy_ready"
    except Exception as e:
        components["vector_store"] = f"error: {e}"

    # Sentiment classifier status
    components["bert_model"] = "keyword_classifier_ready"

    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0",
        components=components,
    )


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("ENVIRONMENT", "production").lower() != "production"

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=port,
        reload=reload,
    )
