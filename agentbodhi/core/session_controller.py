from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .models import AnalysisReport
from .orchestrator import ProgressCallback, ResearchOrchestrator


@dataclass
class AnalysisSession:
    report: AnalysisReport


class SessionController:
    """Keeps the orchestrator, report, and chat context in sync between Streamlit runs."""

    def __init__(self) -> None:
        self._orchestrator: Optional[ResearchOrchestrator] = None
        self._keys: Tuple[Optional[str], Optional[str]] = (None, None)
        self._session: Optional[AnalysisSession] = None

    @property
    def report(self) -> Optional[AnalysisReport]:
        return self._session.report if self._session else None

    def ensure_orchestrator(self, gemini_key: str, tavily_key: str) -> ResearchOrchestrator:
        if not gemini_key or not tavily_key:
            raise ValueError("Gemini and Tavily keys are required")

        key_tuple = (gemini_key, tavily_key)
        if self._orchestrator is None or self._keys != key_tuple:
            self._orchestrator = ResearchOrchestrator(gemini_key, tavily_key)
            self._keys = key_tuple
        return self._orchestrator

    def analyze_pdf(
        self,
        gemini_key: str,
        tavily_key: str,
        pdf_file,
        progress_callback: Optional[ProgressCallback],
        max_citations: int,
        glossary_terms: int,
    ) -> AnalysisReport:
        orchestrator = self.ensure_orchestrator(gemini_key, tavily_key)
        report = orchestrator.analyze_paper(
            pdf_file,
            progress_callback=progress_callback,
            max_citations=max_citations,
            glossary_terms=glossary_terms,
        )
        self._session = AnalysisSession(report=report)
        return report

    def chat_with_agent(self, agent_slug: str, instruction: str, usage_hint: str) -> str:
        if not self._orchestrator:
            raise ValueError("Initialize the orchestrator by analyzing a paper first.")
        return self._orchestrator.chat_with_agent(agent_slug, instruction, usage_hint)

    def has_active_report(self) -> bool:
        return self._session is not None

