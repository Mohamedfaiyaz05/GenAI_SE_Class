from pathlib import Path
import sys

import fitz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from scrape_tn_agriculture_schemes import (
    discover_detail_links,
    extract_scheme,
    write_outputs,
)


def test_discover_detail_links_deduplicates_and_stays_on_target_department():
    listing_url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg%3D%3D"
    html = """
    <a href="scheme_details.php?id=MTU2Ng==">Training to Farmers</a>
    <a href="scheme_details.php?id=MTU2Ng==">Training to Farmers duplicate</a>
    <a href="scheme_list.php?dep_id=Mg%3D%3D&page=2">Next</a>
    <a href="scheme_list.php?dep_id=MQ%3D%3D">Different department</a>
    <a href="https://example.com/scheme_details.php?id=1">External</a>
    """

    details, listings = discover_detail_links(html, listing_url)

    assert len(details) == 1
    assert details[0]["listing_title"] == "Training to Farmers"
    assert len(listings) == 1
    assert "page=2" in listings[0]


def test_extract_scheme_preserves_detail_fields_and_source():
    html = """
    <table>
      <tr><td>Agriculture - Farmers Welfare Department</td></tr>
      <tr><td>Concerned Department:</td><td>Agriculture - Farmers Welfare Department</td></tr>
      <tr><td>Scheme Title/Name:</td><td>Training to Farmers</td></tr>
      <tr><td>Funding Pattern:</td><td>Rs. 300 per farmer for 2 days.</td></tr>
      <tr><td>Description:</td><td>Training is offered to maize growers.</td></tr>
    </table>
    """

    scheme = extract_scheme(
        "https://www.tn.gov.in/scheme_details.php?id=MTU2Ng==",
        "Training to Farmers",
        html,
    )

    assert scheme["title"] == "Training to Farmers"
    assert scheme["department"] == "Agriculture - Farmers Welfare Department"
    assert scheme["source_url"].endswith("id=MTU2Ng==")
    assert {"label": "Funding Pattern", "value": "Rs. 300 per farmer for 2 days."} in scheme["fields"]


def test_write_outputs_creates_json_markdown_and_readable_pdf(tmp_path):
    schemes = [
        {
            "title": "Training to Farmers",
            "listing_title": "Training to Farmers",
            "department": "Agriculture - Farmers Welfare Department",
            "source_url": "https://www.tn.gov.in/scheme_details.php?id=MTU2Ng==",
            "sections": [],
            "fields": [
                {"label": "Beneficiaries", "value": "Farmers"},
                {"label": "Funding Pattern", "value": "Rs. 300 per farmer for 2 days."},
            ],
        }
    ]

    json_path, markdown_path, pdf_path = write_outputs(
        schemes,
        [],
        tmp_path,
        "https://www.tn.gov.in/scheme_list.php?dep_id=Mg==",
    )

    assert json_path.is_file()
    assert markdown_path.is_file()
    assert pdf_path.is_file()
    assert "Training to Farmers" in json_path.read_text(encoding="utf-8")
    assert "Funding Pattern" in markdown_path.read_text(encoding="utf-8")
    with fitz.open(pdf_path) as pdf:
        assert "Training to Farmers" in pdf[0].get_text()
        assert "Rs. 300 per farmer" in pdf[0].get_text()
