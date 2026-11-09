
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import requests
from crewai.tools import tool
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

logger = logging.getLogger(__name__)

SERPAPI_URL = "https://serpapi.com/search.json"

# Trusted domains for general medical education
ALLOWED_DOMAINS = (
    "medlineplus.gov",
    "nhs.uk",
    "cdc.gov",
    "who.int",
    "mayoclinic.org",
)


def is_allowed_domain(url: str) -> bool:
    """Check whether a result belongs to an allowed domain."""

    try:
        parsed = urlparse(url)

        if parsed.scheme != "https":
            return False

        hostname = (parsed.hostname or "").lower()

        return any(
            hostname == domain or hostname.endswith("." + domain)
            for domain in ALLOWED_DOMAINS
        )

    except ValueError:
        return False


@tool("medical_web_search")
def medical_web_search(question: str) -> str:
    """
    Search trusted public health-information websites using SerpAPI.

    Use only for general medical education when internal
    knowledge is insufficient.

    Never use for:
    - Patient records or personal medical information
    - Clinic-specific policies or refund status
    - Medical emergencies
    - Personalized diagnosis or treatment decisions

    Returns search result titles, URLs and snippets.
    Search results are not independently verified.
    """

    if not isinstance(question, str) or not question.strip():
        return json.dumps({
            "status": "invalid_input",
            "error": "Question cannot be empty",
            "results": []
        })

    api_key = os.getenv("SERPAPI_API_KEY")

    if not api_key:
        logger.error("SERPAPI_API_KEY is not configured")

        return json.dumps({
            "status": "configuration_error",
            "error": "Medical web search is unavailable",
            "results": []
        })

    # Restrict search results to approved websites
    site_filter = " OR ".join(
        f"site:{domain}" for domain in ALLOWED_DOMAINS
    )

    params = {
        "engine": "google",
        "q": f"{question.strip()} ({site_filter})",
        "api_key": api_key,
        "num": 10,
        "hl": "en"
    }

    try:
        response = requests.get(
            SERPAPI_URL,
            params=params,
            timeout=(5, 20)
        )

        response.raise_for_status()
        data = response.json()

        if data.get("error"):
            raise RuntimeError("SerpAPI returned an API error")

        results = []

        for item in data.get("organic_results", []):

            url = item.get("link", "")

            if not is_allowed_domain(url):
                continue

            results.append({
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", "")
            })

            if len(results) >= 5:
                break

        return json.dumps({
            "status": "success" if results else "no_results",
            "question": question,
            "result_count": len(results),
            "results": results
        }, ensure_ascii=False)

    except (requests.RequestException, ValueError, RuntimeError):
        logger.exception("Medical web search failed")

        return json.dumps({
            "status": "search_error",
            "error": "Unable to retrieve medical web search results",
            "results": []
        })
