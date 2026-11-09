"""Scrape scheme detail pages linked from the Tamil Nadu agriculture listing."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any
from urllib.parse import parse_qs, urljoin, urlsplit

import fitz
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LIST_URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "scraped_output"
SITE_HOST = "www.tn.gov.in"
USER_AGENT = "VivasayaThozanSchemeArchive/1.0 (polite public-page scraper)"


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


class SchemePageParser(HTMLParser):
    """Collect scheme links and table rows without assuming a page-specific library."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[dict[str, str]] = []
        self.table_rows: list[list[str]] = []
        self._anchor: dict[str, str] | None = None
        self._table_depth = 0
        self._reading_row = False
        self._reading_cell = False
        self._cell_text: list[str] = []
        self._row_cells: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "a":
            self._anchor = {"href": attributes.get("href") or "", "text": ""}
        elif tag == "table":
            self._table_depth += 1
        elif tag == "tr" and self._table_depth == 1:
            self._reading_row = True
            self._row_cells = []
        elif tag in {"td", "th"} and self._reading_row:
            self._reading_cell = True
            self._cell_text = []

    def handle_data(self, data: str) -> None:
        if self._anchor is not None:
            self._anchor["text"] += data
        if self._reading_cell:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._anchor is not None:
            self._anchor["text"] = normalize_text(self._anchor["text"])
            self.links.append(self._anchor)
            self._anchor = None
        elif tag in {"td", "th"} and self._reading_cell:
            self._row_cells.append(normalize_text(" ".join(self._cell_text)))
            self._reading_cell = False
        elif tag == "tr" and self._reading_row:
            if any(self._row_cells):
                self.table_rows.append(self._row_cells)
            self._reading_row = False
        elif tag == "table":
            self._table_depth = max(0, self._table_depth - 1)


def parse_html(html: str) -> SchemePageParser:
    parser = SchemePageParser()
    parser.feed(html)
    parser.close()
    return parser


def is_allowed_site_url(url: str) -> bool:
    parsed = urlsplit(url)
    return parsed.scheme == "https" and parsed.hostname == SITE_HOST


def discover_detail_links(
    html: str,
    listing_url: str,
) -> tuple[list[dict[str, str]], list[str]]:
    parser = parse_html(html)
    detail_links = []
    listing_links = []
    seen_details: set[str] = set()
    seen_listings: set[str] = set()
    root_department = parse_qs(urlsplit(listing_url).query).get("dep_id", [])

    for link in parser.links:
        href = link["href"].strip()
        if not href:
            continue
        absolute_url = urljoin(listing_url, href)
        parsed = urlsplit(absolute_url)
        if not is_allowed_site_url(absolute_url):
            continue

        if parsed.path.rstrip("/").endswith("/scheme_details.php"):
            if absolute_url in seen_details:
                continue
            seen_details.add(absolute_url)
            detail_links.append({
                "url": absolute_url,
                "listing_title": link["text"],
            })
        elif parsed.path.rstrip("/").endswith("/scheme_list.php"):
            child_department = parse_qs(parsed.query).get("dep_id", [])
            if root_department and child_department != root_department:
                continue
            if absolute_url not in seen_listings:
                seen_listings.add(absolute_url)
                listing_links.append(absolute_url)

    return detail_links, listing_links


def extract_scheme(detail_url: str, listing_title: str, html: str) -> dict[str, Any]:
    parser = parse_html(html)
    fields: list[dict[str, str]] = []
    sections: list[str] = []

    for row in parser.table_rows:
        values = [normalize_text(value) for value in row]
        if len(values) == 1:
            if values[0]:
                sections.append(values[0])
            continue

        label = values[0].rstrip(":").strip()
        value = normalize_text(" ".join(values[1:]))
        if label and value:
            fields.append({"label": label, "value": value})

    title = next(
        (
            field["value"]
            for field in fields
            if field["label"].casefold() in {"scheme title/name", "scheme title", "scheme name"}
        ),
        listing_title or "Untitled scheme",
    )
    department = next(
        (
            field["value"]
            for field in fields
            if field["label"].casefold() == "concerned department"
        ),
        "",
    )

    return {
        "title": title,
        "listing_title": listing_title,
        "department": department,
        "source_url": detail_url,
        "sections": sections,
        "fields": fields,
    }


def fetch_html(session: requests.Session, url: str) -> str:
    if not is_allowed_site_url(url):
        raise ValueError(f"Refusing to fetch a URL outside https://{SITE_HOST}: {url}")
    response = session.get(url, timeout=(10, 45))
    response.raise_for_status()
    response.encoding = response.apparent_encoding or response.encoding
    return response.text


def scrape_schemes(
    list_url: str,
    delay_seconds: float = 0.75,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if not is_allowed_site_url(list_url):
        raise ValueError(f"Listing URL must use https://{SITE_HOST}.")
    if delay_seconds < 0:
        raise ValueError("Delay must be zero or greater.")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    listing_queue = [list_url]
    visited_listings: set[str] = set()
    visited_details: set[str] = set()
    schemes: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    while listing_queue:
        current_listing = listing_queue.pop(0)
        if current_listing in visited_listings:
            continue
        visited_listings.add(current_listing)
        try:
            listing_html = fetch_html(session, current_listing)
        except (requests.RequestException, ValueError) as exc:
            failures.append({"url": current_listing, "error": str(exc)})
            continue

        detail_links, next_listings = discover_detail_links(
            listing_html,
            current_listing,
        )
        listing_queue.extend(
            url for url in next_listings
            if url not in visited_listings and url not in listing_queue
        )

        for detail in detail_links:
            detail_url = detail["url"]
            if detail_url in visited_details:
                continue
            visited_details.add(detail_url)
            time.sleep(delay_seconds)
            try:
                detail_html = fetch_html(session, detail_url)
            except (requests.RequestException, ValueError) as exc:
                failures.append({"url": detail_url, "error": str(exc)})
                continue
            schemes.append(
                extract_scheme(detail_url, detail["listing_title"], detail_html)
            )
            print(f"[{len(schemes)}] {schemes[-1]['title']}")

    if not schemes:
        raise RuntimeError(
            "No scheme detail pages were scraped. Check the listing URL and network access."
        )
    return schemes, failures


def scheme_text(scheme: dict[str, Any]) -> str:
    lines = [
        f"Scheme: {scheme['title']}",
        f"Department: {scheme['department'] or 'Not stated'}",
        f"Source page: {scheme['source_url']}",
    ]
    lines.extend(scheme.get("sections", []))
    for field in scheme["fields"]:
        lines.append(f"{field['label']}: {field['value']}")
    return "\n".join(lines)


def wrap_text(
    text: str,
    font_name: str,
    font_size: float,
    max_width: float,
) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if fitz.get_text_length(candidate, fontname=font_name, fontsize=font_size) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def write_pdf(schemes: list[dict[str, Any]], output_path: Path) -> None:
    pdf = fitz.open()
    page_width, page_height = fitz.paper_rect("a4").width, fitz.paper_rect("a4").height
    margin = 48
    body_width = page_width - (margin * 2)
    line_gap = 14

    def add_page(title: str, continuation: bool = False):
        page = pdf.new_page(width=page_width, height=page_height)
        heading = f"{title} (continued)" if continuation else title
        page.insert_text(
            (margin, margin + 16),
            heading,
            fontname="hebo",
            fontsize=15 if not continuation else 12,
        )
        return page, margin + (42 if not continuation else 35)

    for scheme in schemes:
        page, y = add_page(scheme["title"])

        def ensure_space(required: float) -> None:
            nonlocal page, y
            if y + required > page_height - margin:
                page, y = add_page(scheme["title"], continuation=True)
                page.insert_text(
                    (margin, y),
                    f"Source: {scheme['source_url']}",
                    fontname="helv",
                    fontsize=8,
                )
                y += line_gap * 1.5

        for section in scheme.get("sections", []):
            lines = wrap_text(section, "hebo", 10, body_width)
            ensure_space(line_gap * len(lines))
            for line in lines:
                page.insert_text((margin, y), line, fontname="hebo", fontsize=10)
                y += line_gap

        for field in scheme["fields"]:
            label_lines = wrap_text(
                f"{field['label']}:",
                "hebo",
                9,
                body_width,
            )
            value_lines = wrap_text(field["value"], "helv", 9, body_width)
            ensure_space(line_gap * (len(label_lines) + len(value_lines)) + 4)
            for line in label_lines:
                page.insert_text((margin, y), line, fontname="hebo", fontsize=9)
                y += line_gap
            for line in value_lines:
                page.insert_text((margin + 10, y), line, fontname="helv", fontsize=9)
                y += line_gap
            y += 4

        ensure_space(line_gap * 2)
        page.insert_text(
            (margin, y),
            f"Source: {scheme['source_url']}",
            fontname="helv",
            fontsize=8,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".pdf",
        prefix=f".{output_path.stem}-",
        dir=output_path.parent,
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
    try:
        pdf.save(temporary_path)
        os.replace(temporary_path, output_path)
    finally:
        pdf.close()
        temporary_path.unlink(missing_ok=True)


def write_outputs(
    schemes: list[dict[str, Any]],
    failures: list[dict[str, str]],
    output_dir: Path,
    list_url: str,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scraped_at = datetime.now(timezone.utc).isoformat()
    json_path = output_dir / "tn_agriculture_schemes.json"
    markdown_path = output_dir / "tn_agriculture_schemes.md"
    pdf_path = output_dir / "tn_agriculture_schemes.pdf"
    payload = {
        "source_listing": list_url,
        "scraped_at": scraped_at,
        "scheme_count": len(schemes),
        "failed_pages": failures,
        "schemes": schemes,
    }

    json_tmp = json_path.with_suffix(".json.tmp")
    json_tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(json_tmp, json_path)

    markdown_sections = [
        "# Tamil Nadu Agriculture Schemes",
        "",
        f"Source listing: {list_url}",
        f"Scraped at: {scraped_at}",
        f"Scheme pages captured: {len(schemes)}",
        "",
    ]
    for scheme in schemes:
        markdown_sections.extend(
            [
                f"## {scheme['title']}",
                "",
                f"- **Department:** {scheme['department'] or 'Not stated'}",
                f"- **Source:** {scheme['source_url']}",
                "",
            ]
        )
        markdown_sections.extend(
            f"- **{field['label']}:** {field['value']}"
            for field in scheme["fields"]
        )
        markdown_sections.extend(["", "---", ""])
    if failures:
        markdown_sections.extend(["## Pages that could not be fetched", ""])
        markdown_sections.extend(f"- {item['url']}: {item['error']}" for item in failures)
    markdown_path.write_text("\n".join(markdown_sections), encoding="utf-8")
    write_pdf(schemes, pdf_path)
    return json_path, markdown_path, pdf_path


def main() -> int:
    argument_parser = argparse.ArgumentParser(
        description=(
            "Scrape all scheme detail pages linked from the Tamil Nadu "
            "Agriculture - Farmers Welfare listing."
        )
    )
    argument_parser.add_argument(
        "--url",
        default=DEFAULT_LIST_URL,
        help="Agriculture scheme-list URL (default: the supplied department listing).",
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Folder for JSON, Markdown, and RAG-ready PDF output.",
    )
    argument_parser.add_argument(
        "--delay",
        type=float,
        default=0.75,
        help="Pause between detail-page requests in seconds (default: 0.75).",
    )
    args = argument_parser.parse_args()

    try:
        schemes, failures = scrape_schemes(args.url, delay_seconds=args.delay)
        json_path, markdown_path, pdf_path = write_outputs(
            schemes,
            failures,
            args.output_dir,
            args.url,
        )
    except (OSError, requests.RequestException, RuntimeError, ValueError) as exc:
        print(f"Scrape failed: {exc}", file=sys.stderr)
        return 1

    print(f"\nSaved {len(schemes)} scheme records:")
    print(f"  JSON:     {json_path}")
    print(f"  Markdown: {markdown_path}")
    print(f"  PDF:      {pdf_path}")
    if failures:
        print(
            f"Warning: {len(failures)} page(s) could not be fetched; "
            "see failed_pages in the JSON file."
        )
        return 1
    print(
        "\nThe PDF can be uploaded in the app and processed with "
        "'Create chunks & add to vector store'."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
