import os
import re
import json
import asyncio
from typing import Any
from dotenv import load_dotenv

from agents import Agent, AsyncOpenAI, OpenAIChatCompletionsModel, Runner
from agents.run import RunConfig

# Setup
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set in the environment variables.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.5-flash",
    openai_client=external_client,
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True,
)

# Robust extractor
def extract_text(run_result: Any) -> str:
    """
    Robustly extract the human-readable text reply from various RunResult shapes.
    Tries common attributes (output_text, output, final_output, text, messages),
    then falls back to parsing the string representation (looking for "Final output (str):").
    """
    if run_result is None:
        return ""

    # 1) Common direct attrs
    candidate_attrs = [
        "output_text",
        "final_output_text",
        "final_output",
        "output",
        "text",
        "reply",
        "result",
        "response",
    ]
    for attr in candidate_attrs:
        if hasattr(run_result, attr):
            val = getattr(run_result, attr)
            if isinstance(val, str) and val.strip():
                return val.strip()
            # dict-like
            if isinstance(val, dict):
                # Prefer common keys
                for k in ("text", "content", "message", "reply"):
                    if k in val and isinstance(val[k], str) and val[k].strip():
                        return val[k].strip()
                # fallback to JSON string of dict
                try:
                    s = json.dumps(val)
                    if s:
                        return s
                except Exception:
                    pass
            # list-like
            if isinstance(val, (list, tuple)):
                texts = []
                for item in val:
                    if isinstance(item, str) and item.strip():
                        texts.append(item.strip())
                    elif isinstance(item, dict):
                        for k in ("text", "content", "message"):
                            if k in item and isinstance(item[k], str) and item[k].strip():
                                texts.append(item[k].strip())
                if texts:
                    return "\n".join(texts)

    # 2) messages-like structures (list of dicts or objects)
    if hasattr(run_result, "messages"):
        msgs = getattr(run_result, "messages")
        try:
            texts = []
            for m in msgs:
                # dict-style message
                if isinstance(m, dict):
                    # m['content'] may be list/dict/string
                    content = m.get("content", None)
                    if isinstance(content, str) and content.strip():
                        texts.append(content.strip())
                    elif isinstance(content, list):
                        for c in content:
                            if isinstance(c, dict):
                                if "text" in c and isinstance(c["text"], str):
                                    texts.append(c["text"].strip())
                                elif "content" in c and isinstance(c["content"], str):
                                    texts.append(c["content"].strip())
                            elif isinstance(c, str) and c.strip():
                                texts.append(c.strip())
                else:
                    # object-style message: try .content or .text
                    if hasattr(m, "content"):
                        c = getattr(m, "content")
                        if isinstance(c, str) and c.strip():
                            texts.append(c.strip())
                        elif isinstance(c, list):
                            for cc in c:
                                if isinstance(cc, dict) and "text" in cc:
                                    texts.append(cc["text"].strip())
                                elif isinstance(cc, str) and cc.strip():
                                    texts.append(cc.strip())
                    elif hasattr(m, "text"):
                        t = getattr(m, "text")
                        if isinstance(t, str) and t.strip():
                            texts.append(t.strip())
            if texts:
                return "\n".join(texts)
        except Exception:
            pass

    # 3) If run_result itself is a string representation that contains the final output section,
    #    parse the "Final output (str):" block from str(run_result)
    try:
        s = str(run_result)
        # Attempt to capture the multi-line final output block
        m = re.search(r"Final output \\(str\\):\\n\\s*(.+?)(?:\\n- |\\n\\(See|\\Z)", s, flags=re.DOTALL)
        if not m:
            # try without escaping parentheses (some SDKs show 'Final output (str):' plainly)
            m = re.search(r"Final output \(str\):\n\s*(.+?)(?:\n- |\n\(See|$)", s, flags=re.DOTALL)
        if not m:
            # try other label
            m = re.search(r"Final output:\n\s*(.+?)(?:\n- |\n\(See|$)", s, flags=re.DOTALL)
        if m:
            text = m.group(1).strip()
            # Clean repeated lines or trailing metadata markers
            # remove leading/trailing markers
            text = re.sub(r"\n{2,}", "\n\n", text)  # normalize blank lines
            return text
    except Exception:
        pass

    # 4) Last resort: try converting run_result to JSON if possible
    try:
        return json.dumps(run_result)[:10000]
    except Exception:
        pass

    # 5) Final fallback: str()
    return str(run_result).strip()

# Agents
triage_agent = Agent(
    name="Triage Agent",
    instructions="""
        You are a triage assistant. Your job is to classify user issues into one of three buckets:
        - billing: transactions, payments, invoices, subscriptions
        - technical: crashes, bugs, errors, installations
        - general: product usage, features, or other non-billing/technical issues
        Output ONLY one word (billing, technical, or general). If unsure, output 'general'.
    """,
    model=model,
)

billing_agent = Agent(
    name="Billing Support Agent",
    instructions="""
        You are a billing support agent. The user asks about transactions, charges, refunds or account billing.
        IMPORTANT: You do NOT have direct access to user bank or account; if the user asks for personal data, explain the limitation
        and provide instructions to check account/balance via the web portal and steps to contact secure billing support.
        Keep the answer concise and actionable.
    """,
    model=model,
)

technical_agent = Agent(
    name="Technical Support Agent",
    instructions="""
        You are a technical support agent. The user asks about crashes, errors, installation problems, performance issues, etc.
        Ask short clarifying questions if you need more detail; otherwise provide step-by-step troubleshooting.
    """,
    model=model,
)

general_agent = Agent(
    name="General Support Agent",
    instructions="""
        You are a general support agent. Answer product usage or feature questions concisely and include links to docs when helpful.
    """,
    model=model,
)

guardrail_agent = Agent(
    name="Guardrail Agent",
    instructions="""
        You are a guardrail reviewer. INPUT: a candidate reply from another agent.
        Task: If the reply contains disallowed or unsafe content (e.g., reveals private data, uses forbidden words),
        produce a SAFE rewrite of the reply. Otherwise, return the reply UNCHANGED.
        Return only the final reply text (no extra commentary, no metadata).
    """,
    model=model,
)

# Flow
async def run_support_flow(user_input: str, ctx: dict) -> str:
    try:
        # 1) Triage
        triage_run = await Runner.run(
            triage_agent,
            input=user_input,
            run_config=config,
        )
        triage_text = extract_text(triage_run).lower()
        if not triage_text or len(triage_text.split()) > 50 or "runresult" in triage_text.lower():
            # simple keyword fallback
            t = user_input.lower()
            if any(k in t for k in ("refund", "charge", "invoice", "payment", "transaction", "balance", "account")):
                triage_text = "billing"
            elif any(k in t for k in ("crash", "error", "bug", "not open", "freeze", "install", "slow")):
                triage_text = "technical"
            else:
                triage_text = "general"

        # 2) Select agent
        if "bill" in triage_text:
            selected_agent = billing_agent
        elif "tech" in triage_text or "crash" in triage_text or "error" in triage_text:
            selected_agent = technical_agent
        else:
            selected_agent = general_agent

        # 3) Ask chosen agent
        agent_run = await Runner.run(
            selected_agent,
            input=user_input,
            run_config=config,
        )
        agent_reply = extract_text(agent_run)

        if not agent_reply or "runresult" in agent_reply.lower():
            agent_reply = (
                "I cannot access your private account here. "
                "To check your account balance, please sign into the web portal -> Account -> Balance, "
                "or contact billing support at billing@example.com with your account id."
            )

        # 4) Guardrail review (returns final cleaned / approved reply)
        guard_run = await Runner.run(
            guardrail_agent,
            input=agent_reply,
            run_config=config,
        )
        final = extract_text(guard_run)

        if not final or "runresult" in final.lower():
            final = agent_reply

        return final

    except Exception as e:
        return f"An internal error occurred while processing your request. ({e})"

# CLI
def main():
    print("=== Console Support Agent System ===")
    name = input("Your name (optional): ").strip()
    premium = input("Are you a premium user? (y/N): ").strip().lower() == "y"
    ctx = {"name": name, "is_premium": premium}

    print("\nType your issue (type 'exit' to quit).")
    while True:
        user_input = input("\nYou: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print("System: Goodbye!")
            break
        response = asyncio.run(run_support_flow(user_input, ctx))
        print("\nSystem:", response)

if __name__ == "__main__":
    main()
