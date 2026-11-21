import json
import unittest
from unittest.mock import patch

from checker import CheckError
from fallback import check_with_groq_tavily, extract_with_groq

CLAIM = {"claim": "A meeting took place in July 1969.", "page": 1,
         "quote": "A meeting took place in July 1969."}
EVIDENCE = {"results": [{"url": "https://example.org/source", "title": "Archive",
                         "content": "A meeting took place in July 1969 and was recorded."}]}
VERDICT = {"verdict": "Verified", "explanation": "The archive gives the same date.",
           "correction": "", "evidence_id": 1, "evidence_quote": "A meeting took place in July 1969"}


def groq(obj):
    return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(obj)}}]}


class FallbackTests(unittest.TestCase):
    @patch("fallback._post")
    def test_quote_must_be_on_page(self, post):
        post.return_value = groq({"claims": [CLAIM, {**CLAIM, "quote": "Bad quote.", "page": 2}]})
        self.assertEqual(extract_with_groq({1: CLAIM["quote"]}, "example"), [CLAIM])

    @patch("fallback._post")
    def test_url_from_search_not_model(self, post):
        post.side_effect = [EVIDENCE, groq({**VERDICT, "url": "https://example.net/invented"})]
        result = check_with_groq_tavily(CLAIM, "example", "example")
        self.assertEqual(result["sources"], {"https://example.org/source": "Archive"})
        self.assertEqual(post.call_count, 2)

    @patch("fallback._post")
    def test_wrong_quote_does_not_assign_verdict(self, post):
        post.side_effect = [EVIDENCE, groq({**VERDICT, "evidence_quote": "A made-up excerpt."})]
        with self.assertRaises(CheckError):
            check_with_groq_tavily(CLAIM, "example", "example")

    @patch("fallback._post")
    def test_empty_search_has_no_support(self, post):
        post.return_value = {"results": []}
        self.assertEqual(check_with_groq_tavily(CLAIM, "example", "example")["verdict"], "False")
        self.assertEqual(post.call_count, 1)

    @patch("fallback._post")
    def test_outage_does_not_assign_verdict(self, post):
        post.side_effect = CheckError("rate-limited")
        with self.assertRaises(CheckError):
            check_with_groq_tavily(CLAIM, "example", "example")


if __name__ == "__main__":
    unittest.main()
