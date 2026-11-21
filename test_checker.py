import io
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter

from checker import CheckError, check_claim, extract_claims, pdf_pages


def message(text, annotations=()):
    return {"type": "message", "content": [
        {"type": "output_text", "text": text, "annotations": list(annotations)}
    ]}


def response(text, *, web=True, annotations=()):
    search = [{"type": "web_search_call", "status": "completed"}] if web else []
    return {"status": "completed", "output": search + [message(text, annotations)]}


FIXTURE = Path(__file__).with_name("sample-test.pdf")
CLAIM = {"page": 1, "quote": "The James Webb Space Telescope launched on December 25, 2021.",
         "claim": "The James Webb Space Telescope launched on December 25, 2021."}
CITATION = {"type": "url_citation", "url": "https://science.nasa.gov/mission/webb/launch/",
            "title": "NASA Webb launch timeline"}


class CheckerTests(unittest.TestCase):
    def test_pdf_preserves_pages(self):
        pages = pdf_pages(FIXTURE.read_bytes())
        self.assertIn("asteroid Bennu", pages[1])
        self.assertIn("Project Saffron", pages[2])

    def test_blank_and_invalid_pdf_rejected(self):
        output = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=100, height=100)
        writer.write(output)
        with self.assertRaisesRegex(CheckError, "No selectable text"):
            pdf_pages(output.getvalue())
        with self.assertRaisesRegex(CheckError, "valid PDF"):
            pdf_pages(b"not a PDF")

    def test_only_exact_page_quotes_accepted(self):
        data = {"claims": [CLAIM, {**CLAIM, "page": 2}, {**CLAIM, "quote": "Invented text"},
                           CLAIM, None, {}, {"page": True}]}
        result = extract_claims(pdf_pages(FIXTURE.read_bytes()), "unused",
                                call=lambda *_: response(json.dumps(data), web=False))
        self.assertEqual(result, [CLAIM])

    def test_verified_with_source(self):
        def stub(*_, **kwargs):
            self.assertTrue(kwargs["web"])
            return response("VERIFIED\nEvidence: NASA reports a 2021-12-25 launch.",
                            annotations=[CITATION])
        result = check_claim(CLAIM, "unused", call=stub)
        self.assertEqual(result["verdict"], "Verified")
        self.assertIn(CITATION["url"], result["sources"])

    def test_inaccurate_returns_correction(self):
        result = check_claim(CLAIM, "unused", call=lambda *_, **__: response(
            "INACCURATE\nEvidence: Date doesn't match.\nCorrected fact: December 25, 2021.",
            annotations=[CITATION]))
        self.assertEqual(result["verdict"], "Inaccurate")
        self.assertEqual(result["correction"], "December 25, 2021.")

    def test_citationless_or_failed_search_is_not_a_verdict(self):
        for stub_result in [response("VERIFIED\nEvidence: True."),
                            response("FALSE\nEvidence: None.", web=False)]:
            with self.assertRaises(CheckError):
                check_claim(CLAIM, "unused", call=lambda *_, **__: stub_result)

    def test_no_evidence_does_not_invent_correction(self):
        result = check_claim(CLAIM, "unused", call=lambda *_, **__: response(
            "FALSE\nEvidence: No evidence.\nCorrected fact: Fake correction."))
        self.assertEqual(result["verdict"], "False")
        self.assertEqual(result["correction"], "")

    def test_request_web_is_required(self):
        with patch("checker.urlopen") as request:
            request.return_value.__enter__.return_value.read.return_value = json.dumps(
                response("VERIFIED\nEvidence: NASA.", annotations=[CITATION])).encode()
            check_claim(CLAIM, "test")
            sent = json.loads(request.call_args[0][0].data)
            self.assertEqual(sent["tools"], [{"type": "web_search", "external_web_access": True}])
            self.assertEqual(sent["tool_choice"], "required")
            self.assertEqual(sent["store"], False)
