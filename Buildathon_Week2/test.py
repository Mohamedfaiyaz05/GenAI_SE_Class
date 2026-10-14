import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import requests


def recursive_web_crawler(start_url: str, max_depth: int = 3, max_pages: int = 50):
    """Recursively crawls a website starting from start_url, extracting text content

    and discovering child URLs within the same domain.
    """
    visited_urls = set()
    queue = [(start_url, 0)]  # Queue stores tuples of (url, current_depth)
    scraped_documents = []

    # Extract base domain to restrict crawling to the target site
    parsed_start = urlparse(start_url)
    base_domain = parsed_start.netloc

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    print(
        f"Starting recursive crawl on domain: {base_domain} (Max Depth: {max_depth})"
    )

    while queue and len(visited_urls) < max_pages:
        current_url, depth = queue.pop(0)

        # Normalize URL by removing fragments (#anchor)
        current_url = current_url.split("#")[0]

        if current_url in visited_urls or depth > max_depth:
            continue

        print(
            f"[{len(visited_urls) + 1}] Scraping (Depth {depth}): {current_url}"
        )
        visited_urls.add(current_url)

        try:
            response = requests.get(
                current_url, headers=headers, timeout=10, allow_redirects=True
            )
            if response.status_code != 200:
                print(f"  -> Skipped: Status code {response.status_code}")
                continue

            # Ensure we only process HTML content
            content_type = response.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue

            soup = BeautifulSoup(response.text, "html.parser")

            # 1. Clean boilerplate elements to keep text clean for RAG
            for element in soup(
                ["script", "style", "nav", "footer", "header", "noscript"]
            ):
                element.decompose()

            # Extract clean text
            text_content = soup.get_text(separator="\n", strip=True)
            if text_content:
                scraped_documents.append(
                    {
                        "url": current_url,
                        "depth": depth,
                        "content": text_content,
                    }
                )

            # 2. Extract child URLs if we haven't hit max depth
            if depth < max_depth:
                for link in soup.find_all("a", href=True):
                    raw_href = link["href"]
                    absolute_url = urljoin(current_url, raw_href)
                    parsed_target = urlparse(absolute_url)

                    # --- Strict Filtering Rules ---
                    # Rule A: Must belong to the same domain
                    # Rule B: Must be http or https
                    # Rule C: Exclude non-document file types (images, archives, scripts)
                    # Rule D: Must not already be visited
                    if (
                        parsed_target.netloc == base_domain
                        and parsed_target.scheme in ["http", "https"]
                        and not any(
                            absolute_url.lower().endswith(ext)
                            for ext in [
                                ".pdf",
                                ".png",
                                ".jpg",
                                ".jpeg",
                                ".zip",
                                ".css",
                                ".js",
                                ".svg",
                                ".xml",
                            ]
                        )
                    ):
                        clean_target_url = absolute_url.split("#")[0]
                        if clean_target_url not in visited_urls:
                            queue.append((clean_target_url, depth + 1))

            # Politeness delay to avoid overloading the target server
            time.sleep(1)

        except Exception as e:
            print(f"  -> Error scraping {current_url}: {e}")

    print(
        f"\nCrawl complete. Successfully scraped {len(scraped_documents)} pages."
    )
    return scraped_documents