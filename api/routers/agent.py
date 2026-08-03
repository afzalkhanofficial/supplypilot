"""
Agent and risk-alert routes.

Endpoints
---------
POST /agent/chat            — Submit a question to the SupplyPilot agent.
GET  /agent/history         — Retrieve past agent interactions.
GET  /alerts                — List recent risk alerts.
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import text

from api.schemas import (
    AgentChatRequest,
    AgentChatResponse,
    AgentHistoryResponse,
    AgentInteractionOut,
    RiskAlertListResponse,
    RiskAlertOut,
)

logger = logging.getLogger(__name__)

agent_router = APIRouter(prefix="/agent", tags=["Agent"])
alerts_router = APIRouter(prefix="/alerts", tags=["Risk Alerts"])


def _engine():
    from database.db import engine
    return engine


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------

@agent_router.post(
    "/chat",
    response_model=AgentChatResponse,
    summary="Chat with the SupplyPilot AI agent",
)
def agent_chat(body: AgentChatRequest):
    """
    Send a natural-language question to the SupplyPilot agent and receive
    a structured response.

    The agent will call whichever combination of tools it needs
    (forecasting, inventory, purchase orders, alerts) and return a
    concise, data-backed answer.

    Optionally pass ``chat_history`` to maintain multi-turn context
    across requests.
    """
    from agent.agent import build_chat_history, run_agent

    history = build_chat_history(
        [(t.human, t.ai) for t in body.chat_history]
    ) if body.chat_history else []

    try:
        result = run_agent(body.question, chat_history=history)
    except EnvironmentError as exc:
        # Missing GROQ_API_KEY — surface as 503 rather than 500.
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        logger.exception("agent_chat: unhandled error")
        raise HTTPException(status_code=500, detail=str(exc))

    return AgentChatResponse(
        answer=result["answer"],
        tools_used=result["tools_used"],
        steps=result["steps"],
    )


@agent_router.get(
    "/history",
    response_model=AgentHistoryResponse,
    summary="Retrieve past agent interactions",
)
def agent_history(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Return logged agent interactions, newest first."""
    with _engine().connect() as conn:
        total = conn.execute(
            text("SELECT COUNT(*) FROM agent_interactions")
        ).scalar()

        rows = conn.execute(
            text("""
                SELECT id, user_question, agent_answer, tools_used, created_at
                FROM   agent_interactions
                ORDER  BY created_at DESC
                LIMIT  :lim OFFSET :off
            """),
            {"lim": limit, "off": offset},
        ).fetchall()

    interactions = [
        AgentInteractionOut(
            id=r[0],
            user_question=r[1],
            agent_answer=r[2],
            tools_used=r[3],
            created_at=r[4],
        )
        for r in rows
    ]
    return AgentHistoryResponse(interactions=interactions, total=int(total))


# ---------------------------------------------------------------------------
# Risk alert endpoints
# ---------------------------------------------------------------------------

@alerts_router.get(
    "",
    response_model=RiskAlertListResponse,
    summary="List recent risk alerts",
)
def list_alerts(
    severity: Optional[str] = Query(
        default=None,
        description="Filter by severity: 'low', 'medium', or 'high'.",
        pattern="^(low|medium|high)$",
    ),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    """Return risk alerts ordered by created_at descending."""
    base = """
        SELECT id, product_id, alert_type, message, severity, created_at
        FROM   risk_alerts
    """
    count_base = "SELECT COUNT(*) FROM risk_alerts"

    params: dict = {"lim": limit, "off": offset}
    where = ""
    if severity:
        where = " WHERE severity = :severity"
        params["severity"] = severity

    with _engine().connect() as conn:
        total = conn.execute(text(count_base + where), params).scalar()
        rows = conn.execute(
            text(base + where + " ORDER BY created_at DESC LIMIT :lim OFFSET :off"),
            params,
        ).fetchall()

    alerts = [
        RiskAlertOut(
            id=r[0],
            product_id=r[1],
            alert_type=r[2],
            message=r[3],
            severity=r[4],
            created_at=r[5],
        )
        for r in rows
    ]
    return RiskAlertListResponse(alerts=alerts, total=int(total))
