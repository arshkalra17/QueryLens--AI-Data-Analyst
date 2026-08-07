"""Streamlit UI for the AI Data Analyst.

A thin client of the FastAPI REST API (backend running on :8000). Handles:
upload -> session -> chat (streaming) -> charts (interactive Plotly via the
chart store) -> "show reasoning" toggle (retrieved context, tools, SQL).

Run:  streamlit run app/ui/streamlit_app.py  (from backend/)
"""
from __future__ import annotations

import json
import os

import httpx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

API_BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "dataset_ids" not in st.session_state:
    st.session_state.dataset_ids = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "show_reasoning" not in st.session_state:
    st.session_state.show_reasoning = False


# ---------------------------------------------------------------- api helpers
def _headers():
    return {"Content-Type": "application/json"}


def api_get(path: str) -> dict:
    with httpx.Client(timeout=30) as client:
        resp = client.get(API_BASE + path)
        resp.raise_for_status()
        return resp.json()


def ensure_session() -> str:
    if st.session_state.session_id:
        return st.session_state.session_id
    with httpx.Client(timeout=30) as client:
        resp = client.post(API_BASE + "/api/chat/session", json={"dataset_ids": st.session_state.dataset_ids})
        resp.raise_for_status()
        sid = resp.json()["session_id"]
    st.session_state.session_id = sid
    return sid


def upload_files(files: list) -> list[str]:
    ids = []
    with httpx.Client(timeout=60) as client:
        for f in files:
            resp = client.post(
                API_BASE + "/api/datasets",
                files={"files": (f.name, f.getvalue(), "text/csv")},
            )
            data = resp.json()
            for imp in data.get("imported", []):
                ids.append(imp["dataset_id"])
            for err in data.get("errors", []):
                st.error(f"{err['file']}: {err['error']}")
    st.session_state.dataset_ids = list(dict.fromkeys(st.session_state.dataset_ids + ids))
    return ids


def refresh_datasets() -> list[dict]:
    data = api_get("/api/datasets")
    st.session_state.dataset_ids = [d["dataset_id"] for d in data["datasets"]]
    return data["datasets"]


def stream_chat(question: str) -> dict:
    """POST /api/chat (blocking JSON) and return the structured response."""
    with httpx.Client(timeout=180) as client:
        resp = client.post(
            API_BASE + "/api/chat",
            json={
                "session_id": st.session_state.session_id,
                "message": question,
                "dataset_ids": st.session_state.dataset_ids,
            },
        )
        resp.raise_for_status()
        return resp.json()


def get_plotly_spec(chart_id: str) -> dict | None:
    try:
        return api_get(f"/api/charts/{chart_id}/plotly").get("spec")
    except Exception:
        return None


def render_markdown_safe(text: str) -> str:
    """Minimal renderer: escape, then re-enable image tags pointing at the API."""
    import html

    escaped = html.escape(text or "")
    return escaped


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("📊 AI Data Analyst")
    st.caption("RAG + Groq · no raw data dumped into the prompt")
    st.divider()

    uploaded = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)
    if uploaded:
        with st.spinner("Validating + indexing…"):
            new_ids = upload_files(uploaded)
        if new_ids:
            st.success(f"Imported {len(new_ids)} dataset(s)")

    if st.button("Load sample datasets"):
        with st.spinner("Loading samples…"):
            for name in ["sales_data.csv", "retail_transactions.csv", "web_traffic.csv"]:
                resp = httpx.get(f"{API_BASE}/data/samples/{name}")
                if resp.status_code == 200:
                    import io

                    f = io.BytesIO(resp.content)
                    f.name = name
                    upload_files([f])
        st.rerun()

    st.divider()
    datasets = refresh_datasets()
    if datasets:
        st.markdown("**Loaded datasets**")
        for d in datasets:
            st.markdown(f"- `{d['name']}` — {d['row_count']} rows × {d['col_count']} cols")

    st.session_state.show_reasoning = st.toggle("Show reasoning (retrieved context, tools, SQL)", value=st.session_state.show_reasoning)

    if st.button("🔄 New chat"):
        st.session_state.session_id = None
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------- main chat
st.subheader("Ask your data anything")

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m.get("content", ""))
        for chart_id in m.get("chart_ids", []):
            spec = get_plotly_spec(chart_id)
            if spec:
                st.plotly_chart(go.Figure(spec), use_container_width=True)
        if st.session_state.show_reasoning and m.get("reasoning"):
            with st.expander("🧠 Reasoning & retrieved context"):
                st.markdown(m["reasoning"])

question = st.chat_input("e.g. Which region generated the highest revenue?")
if question:
    if not st.session_state.dataset_ids:
        st.warning("Upload a CSV first.")
        st.stop()
    ensure_session()
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question, "chart_ids": []})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_analysing…_")
        try:
            payload = stream_chat(question)
        except Exception as exc:
            placeholder.error(f"Request failed: {exc}")
            st.stop()

        answer = payload.get("answer", "")
        chart_ids = payload.get("chart_ids", [])
        reasoning = payload.get("reasoning", "")
        sql = payload.get("sql")
        tools = payload.get("tools", [])

        placeholder.markdown(answer)
        for chart_id in chart_ids:
            spec = get_plotly_spec(chart_id)
            if spec:
                st.plotly_chart(go.Figure(spec), use_container_width=True)
            else:
                st.image(f"{API_BASE}/api/charts/{chart_id}/image", use_container_width=True)

        if st.session_state.show_reasoning:
            with st.expander("🧠 Reasoning & retrieved context"):
                if tools:
                    st.markdown("**Tools called:** " + ", ".join(f"`{t['name']}`" for t in tools))
                if reasoning:
                    st.markdown(reasoning)
                if sql:
                    st.code(sql, language="sql")
                meta = payload.get("provider", "")
                st.caption(f"provider: {meta}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "chart_ids": chart_ids, "reasoning": reasoning}
        )
