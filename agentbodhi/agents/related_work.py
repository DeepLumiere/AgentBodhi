import logging
import time

import arxiv

from ..core.utils import clean_query, shared_arxiv_client
from .base import ResearchAgent

logger = logging.getLogger(__name__)


class RelatedWorkAgent(ResearchAgent):
    """Uses Arxiv primarily for high-quality academic connections."""

    def execute(self, paper_summary: str, num_papers: int = 5):
        try:
            query_prompt = f"Extract a concise 3-4 word search query to find related academic papers based on this summary:\n{paper_summary[:500]}"
            query_response = self.client.models.generate_content(
                model=self.model,
                contents=query_prompt
            )
            search_query = clean_query(query_response.text.strip())

            search = arxiv.Search(
                query=f'all:{search_query}',
                max_results=num_papers,
                sort_by=arxiv.SortCriterion.Relevance
            )
            time.sleep(3) # Extra delay protection

            related = []
            for paper in shared_arxiv_client.results(search):
                related.append({
                    'title': paper.title,
                    'url': paper.entry_id,
                    'snippet': paper.summary[:200] + '...',
                    'relevance': 0.9
                })

            return related
        except Exception as e:
            logger.error(f"Related work search error: {e}")
            return []
