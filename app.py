import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests
import streamlit as st


st.set_page_config(
    page_title="Intelligent Annotation Assistant",
    page_icon="🤖",
    layout="wide",
)


DEFAULT_GUIDELINES = """You are an Intelligent Annotation Assistant for AGIDS workflows.

Follow these principles:
1) Be concise, objective, and annotation-focused.
2) Prioritize factual correctness and guideline adherence.
3) Avoid speculation. If uncertain, say what is uncertain.
4) For text tasks: help with grammar, clarity, and fact checks.
5) For audio/video/image tasks: explain unfamiliar entities, actions, or terminology clearly.
6) Keep responses practical so Data Associates can proceed quickly.
"""


WORKFLOW_HELP = {
    "Text": "Focus on grammar, factual checks, phrasing clarity, and ambiguity reduction.",
    "Audio (ASR)": "Focus on validating names, titles, entities, and likely transcription confusions.",
    "Video": "Focus on identifying actions/events and terminology for niche activities.",
    "Image": "Focus on object/scene terminology and prompt interpretation support.",
}


def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except OSError:
        pass


load_env_file()


def provider_defaults(provider: str) -> Dict[str, str]:
    if provider == "Groq":
        return {
            "api_key": os.getenv("GROQ_API_KEY", ""),
            "model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        }
    return {
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "model": os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
    }


def call_groq(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"].strip()


def call_gemini(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    system_parts: List[str] = []
    convo_parts: List[str] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            system_parts.append(content)
        else:
            label = "User" if role == "user" else "Assistant"
            convo_parts.append(f"{label}: {content}")

    prompt = ""
    if system_parts:
        prompt += "System instructions:\n" + "\n".join(system_parts) + "\n\n"
    prompt += "Conversation:\n" + "\n".join(convo_parts)

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    response = requests.post(url, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"Gemini returned no candidates: {json.dumps(data)[:500]}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_chunks = [p.get("text", "") for p in parts if "text" in p]
    answer = "\n".join(text_chunks).strip()
    if not answer:
        raise ValueError(f"Gemini returned empty text: {json.dumps(data)[:500]}")
    return answer


def build_messages(
    guideline_text: str,
    workflow_type: str,
    task_text: str,
    chat_history: List[Dict[str, str]],
) -> List[Dict[str, str]]:
    workflow_hint = WORKFLOW_HELP.get(workflow_type, "")
    system_prompt = (
        f"{guideline_text}\n\n"
        f"Active workflow: {workflow_type}\n"
        f"Workflow hint: {workflow_hint}\n\n"
        "Use this current task context when useful:\n"
        f"{task_text if task_text.strip() else '[No task text provided]'}"
    )
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    return messages


def ask_model(
    provider: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    if provider == "Groq":
        return call_groq(api_key, model, messages, temperature, max_tokens)
    return call_gemini(api_key, model, messages, temperature, max_tokens)


def add_user_message(content: str) -> None:
    st.session_state.chat.append({"role": "user", "content": content})


def add_assistant_message(content: str) -> None:
    st.session_state.chat.append({"role": "assistant", "content": content})


def run_assistant(user_query: str) -> None:
    provider = st.session_state.provider
    api_key = st.session_state.api_key.strip()
    model = st.session_state.model.strip()
    temperature = st.session_state.temperature
    max_tokens = st.session_state.max_tokens

    if not api_key:
        st.error("Add an API key in the sidebar to continue.")
        return
    if not model:
        st.error("Select or enter a model name in the sidebar.")
        return

    add_user_message(user_query)
    messages = build_messages(
        guideline_text=st.session_state.guidelines,
        workflow_type=st.session_state.workflow_type,
        task_text=st.session_state.task_input,
        chat_history=st.session_state.chat,
    )

    try:
        with st.spinner("Assistant is thinking..."):
            answer = ask_model(provider, api_key, model, messages, temperature, max_tokens)
        add_assistant_message(answer)
    except Exception as exc:
        add_assistant_message(f"Error while contacting {provider}: {exc}")


if "chat" not in st.session_state:
    st.session_state.chat = []
if "task_input" not in st.session_state:
    st.session_state.task_input = "Paris is the capital of Frace."
if "workflow_type" not in st.session_state:
    st.session_state.workflow_type = "Text"
if "guidelines" not in st.session_state:
    st.session_state.guidelines = DEFAULT_GUIDELINES
if "provider" not in st.session_state:
    st.session_state.provider = os.getenv("API_PROVIDER", "Groq").strip() or "Groq"
if "api_key" not in st.session_state:
    st.session_state.api_key = provider_defaults(st.session_state.provider)["api_key"]
if "model" not in st.session_state:
    st.session_state.model = provider_defaults(st.session_state.provider)["model"]
if "temperature" not in st.session_state:
    st.session_state.temperature = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
if "max_tokens" not in st.session_state:
    st.session_state.max_tokens = int(os.getenv("MODEL_MAX_TOKENS", "512"))

# Always refresh runtime config from .env (no UI controls for model setup).
st.session_state.provider = os.getenv("API_PROVIDER", st.session_state.provider).strip() or "Groq"
env_defaults = provider_defaults(st.session_state.provider)
st.session_state.api_key = env_defaults["api_key"]
st.session_state.model = env_defaults["model"]
st.session_state.temperature = float(os.getenv("MODEL_TEMPERATURE", str(st.session_state.temperature)))
st.session_state.max_tokens = int(os.getenv("MODEL_MAX_TOKENS", str(st.session_state.max_tokens)))


st.title("Embedded Intelligent Annotation Assistant")
st.caption("Context-aware real-time support for AGIDS-style annotation workflows.")

with st.sidebar:
    st.subheader("Configuration")
    st.caption("Model setup is loaded from `.env`.")
    provider_env = provider_defaults(st.session_state.provider)
    st.session_state.api_key = provider_env["api_key"]
    st.session_state.model = provider_env["model"]
    st.caption(f"Provider: {st.session_state.provider}")
    st.caption(f"Model: {st.session_state.model}")

    if st.button("Clear Chat"):
        st.session_state.chat = []
        st.rerun()


left_col, right_col = st.columns([1.45, 1.0], gap="large")

with left_col:
    st.subheader("Annotation Task Panel")
    st.session_state.workflow_type = st.selectbox("Workflow Type", list(WORKFLOW_HELP.keys()), index=list(WORKFLOW_HELP.keys()).index(st.session_state.workflow_type))
    st.info(f"Workflow guidance: {WORKFLOW_HELP[st.session_state.workflow_type]}")
    st.session_state.task_input = st.text_area(
        "Task content / prompt",
        value=st.session_state.task_input,
        height=180,
        help="Paste a task snippet here. The assistant uses this as context.",
    )

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("Submit Task"):
            st.success("Task submitted (demo action).")
    with b2:
        if st.button("Skip"):
            st.warning("Task skipped (demo action).")
    with b3:
        if st.button("Flag Issue"):
            st.error("Issue flagged (demo action).")

    st.markdown("### Quick Actions")
    q1, q2, q3, q4 = st.columns(4)
    if q1.button("Fact Check"):
        run_assistant("Fact-check this content and list corrections needed.")
        st.rerun()
    if q2.button("Grammar Fix"):
        run_assistant("Correct grammar and spelling only. Keep the original meaning.")
        st.rerun()
    if q3.button("Define Term"):
        run_assistant("Identify uncommon words/entities in the task and define each briefly.")
        st.rerun()
    if q4.button("Guideline Help"):
        run_assistant("What annotation guidelines are most relevant to this task?")
        st.rerun()

    with st.expander("Guidelines / Conventions (editable)"):
        st.session_state.guidelines = st.text_area(
            "Assistant instruction base",
            value=st.session_state.guidelines,
            height=220,
        )

with right_col:
    st.subheader("Intelligent Annotation Assistant")
    st.caption(f"Session started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    chat_box = st.container(height=520, border=True)
    with chat_box:
        if not st.session_state.chat:
            st.markdown(
                "Ask questions like:\n"
                "- `Is this sentence correct?`\n"
                "- `Who is this entity in the audio snippet?`\n"
                "- `What does this sports move mean?`\n"
                "- `Which guideline applies here?`"
            )
        else:
            for msg in st.session_state.chat:
                with st.chat_message("user" if msg["role"] == "user" else "assistant"):
                    st.markdown(msg["content"])

    user_text = st.chat_input("Ask a question...")
    if user_text:
        run_assistant(user_text)
        st.rerun()
