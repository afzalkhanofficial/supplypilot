"""
SupplyPilot agent runner.

Wires together:
  - Groq-hosted LLM (llama-3.3-70b-versatile via langchain-groq)
  - Six supply-chain tools defined in agent.tools
  - The system prompt from agent.prompts
  - A LangChain tool-calling AgentExecutor

Public interface
----------------
    from agent.agent import run_agent

    response = run_agent("Which products are at critical stock risk?")
    print(response["answer"])

The function also logs every interaction (question + answer + tools used)
to the agent_interactions table for audit purposes.

Design notes
------------
- The agent is rebuilt on every call.  This is intentional: it keeps the
  module stateless and avoids stale chat history leaking across dashboard
  sessions.  For multi-turn conversation, callers pass chat_history
  explicitly (see the chat_history parameter).
- Temperature is set to 0 for deterministic, reproducible tool calls.
  Inventory decisions should not vary between equivalent queries.
- max_iterations is capped at 8 to prevent infinite tool-call loops while
  still allowing the agent to chain several lookups.
"""

import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_groq import ChatGroq
from sqlalchemy import text

from agent.prompts import build_prompt
from agent.tools import ALL_TOOLS

load_dotenv()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

_MODEL_NAME = "llama-3.3-70b-versatile"  # Groq's current high-quality fast model
_TEMPERATURE = 0
_MAX_ITERATIONS = 8
_MAX_EXECUTION_TIME = 120  # seconds — hard cap to keep the API responsive


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

def _build_agent_executor() -> AgentExecutor:
    """
    Construct a fresh AgentExecutor with the Groq LLM and all supply-chain
    tools.

    Raises
    ------
    EnvironmentError
        If the GROQ_API_KEY environment variable is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file."
        )

    llm = ChatGroq(
        model=_MODEL_NAME,
        temperature=_TEMPERATURE,
        api_key=api_key,
    )

    prompt = build_prompt()
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)

    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,           # set True locally to see the chain-of-thought
        max_iterations=_MAX_ITERATIONS,
        max_execution_time=_MAX_EXECUTION_TIME,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# ---------------------------------------------------------------------------
# Interaction logger
# ---------------------------------------------------------------------------

def _log_interaction(
    user_question: str,
    agent_answer: str,
    tools_used: list[str],
) -> None:
    """
    Persist the agent interaction to the agent_interactions table.

    Failures are logged as warnings and silently swallowed — a logging
    failure must never take down the agent response.

    Parameters
    ----------
    user_question:
        The raw question sent by the user.
    agent_answer:
        The final answer string returned by the agent.
    tools_used:
        List of tool names called during this interaction.
    """
    from database.db import engine

    tools_str = ", ".join(tools_used) if tools_used else ""
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO agent_interactions
                        (user_question, agent_answer, tools_used, created_at)
                    VALUES (:q, :a, :t, NOW())
                """),
                {"q": user_question, "a": agent_answer, "t": tools_str},
            )
    except Exception:
        logger.warning("Failed to log agent interaction to DB.", exc_info=True)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def run_agent(
    user_question: str,
    chat_history: list[BaseMessage] | None = None,
) -> dict[str, Any]:
    """
    Run the SupplyPilot agent on a single user question and return the
    structured result.

    Parameters
    ----------
    user_question : str
        The natural-language question or instruction from the user.
    chat_history : list[BaseMessage] | None
        Prior messages in the conversation for multi-turn context.
        Each entry should be a LangChain HumanMessage or AIMessage.
        Pass None (the default) for single-turn queries.

    Returns
    -------
    dict with keys:

    - ``answer`` (str): The agent's final natural-language response.
    - ``tools_used`` (list[str]): Names of tools called in this turn.
    - ``steps`` (int): Number of agent iterations used.

    Raises
    ------
    EnvironmentError
        If GROQ_API_KEY is missing.
    """
    executor = _build_agent_executor()

    inputs: dict[str, Any] = {
        "input": user_question,
        "chat_history": chat_history or [],
    }

    try:
        result = executor.invoke(inputs)
    except Exception as exc:
        logger.exception("Agent executor raised an unhandled exception.")
        error_answer = (
            f"The agent encountered an error while processing your request: {exc}. "
            "Please try rephrasing or check that all services are running."
        )
        _log_interaction(user_question, error_answer, [])
        return {"answer": error_answer, "tools_used": [], "steps": 0}

    answer: str = result.get("output", "")
    intermediate_steps = result.get("intermediate_steps", [])

    # Extract the name of every tool that was called.
    tools_used: list[str] = []
    for action, _observation in intermediate_steps:
        tool_name = getattr(action, "tool", None)
        if tool_name and tool_name not in tools_used:
            tools_used.append(tool_name)

    _log_interaction(user_question, answer, tools_used)

    return {
        "answer": answer,
        "tools_used": tools_used,
        "steps": len(intermediate_steps),
    }


def build_chat_history(
    turns: list[tuple[str, str]],
) -> list[BaseMessage]:
    """
    Convert a list of (human_text, ai_text) tuples into LangChain message
    objects for use as chat_history in run_agent().

    Parameters
    ----------
    turns : list of (str, str)
        Each tuple is (user_message, agent_response) in chronological order.

    Returns
    -------
    list[BaseMessage]
        Ready to pass as the chat_history parameter to run_agent().
    """
    messages: list[BaseMessage] = []
    for human_text, ai_text in turns:
        messages.append(HumanMessage(content=human_text))
        messages.append(AIMessage(content=ai_text))
    return messages
