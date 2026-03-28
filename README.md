# Agent Bodhi API Chat

A FastAPI + plain HTML/JS research co-pilot! Upload a PDF inside a chat GUI and select multiple parallel agents (Citations, Methodology, Novelty, etc.) to query them simultaneously.

## Core Features

- **FastAPI Engine**: Powered by Uvicorn, replacing the cumbersome Streamlit flow.
- **Dynamic & Parallel Chat Agents**: Multiselect your preferred agents. All selected bots process your query and relevant paper context at the same time.
- **True Agentic AI (Marionette Search)**: Features autonomous agents like the **Conference Matchmaker**, which intelligently generates search queries, browses the live web via Tavily, and streams a live "Agent Activity Log" of its thoughts and actions before evaluating your paper against real-world Call for Papers (CFP) requirements.
- **HTML/CSS/JS Frontend**: Clean, responsive layout that feels like a native chat application.
- **No API Hurdles in UI**: Configuration implicitly loaded from `config.py`.

## Quick Start

1. Initialize a Python 3.11+ environment.
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies.
   ```powershell
   pip install fastapi uvicorn python-multipart google-generativeai tavily arxiv
   ```
3. Ensure `config.py` has valid API credentials for `GEMINI_API_KEY` and `TAVILY_API_KEY`.
4. Run the server.
   ```powershell
   python app.py
   ```
5. Open `http://127.0.0.1:8000` in your web browser. Upload a paper using the button at the top header, choose your agents from the sidebar, and start chatting.

## File Layout

- `app.py` – FastAPI routes and application assembly.
- `static/index.html` – Lightweight HTML chat interface (with bundled minimal CSS/JS).
- `config.py` – Local secrets module.
- `agentbodhi/` – Backend orchestration logic and agent class definitions.
