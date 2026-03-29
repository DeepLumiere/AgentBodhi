import json
import re

import arxiv

shared_arxiv_client = arxiv.Client(page_size=3, delay_seconds=3, num_retries=1)


def clean_query(q: str) -> str:
    q = re.sub(r"[\*\[\]\(\)\"\'`]", '', q)
    q = re.sub(r'\s+', ' ', q)
    return q.strip()


def extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if match:
        return match.group(0)

    raise ValueError('No valid JSON found')
