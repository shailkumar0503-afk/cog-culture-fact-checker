import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from groq import Groq

from checker import CheckError
from fallback import check_with_groq_tavily, extract_with_groq

CLAIM = {"claim": "A meeting took place in July 1969.", "page": 1,
         "quote": "A meeting took place in July 1969."}
EVIDENCE = {"results": [{"url": "https://example.org/source", "title": "Archive",
                         "content": "A meeting took place in July 1969 and was recorded."}]}
VERDICT = {"verdict": "Verified", "explanation": "The archive gives the same date.",
           "correction": "", "passage_id": 1}


def groq(obj):
    choice = SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=json.dumps(obj)))
    return SimpleNamespace(choices=[choice])


class FallbackTests(unittest.TestCase):
    def test_quote_must_be_on_page(self):
        with patch("fallback.Groq") as client:
            create = client.return_value.__enter__.return_value.chat.completions.create
            create.return_value = groq({"claims": [CLAIM, {**CLAIM, "quote": "Bad quote.", "page": 2}]})
            self.assertEqual(extract_with_groq({1: CLAIM["quote"]}, "example"), [CLAIM])
            self.assertEqual(create.call_args.kwargs["reasoning_effort"], "low")
            fmt = create.call_args.kwargs["response_format"]
            self.assertEqual(fmt["type"], "json_schema")
            self.assertTrue(fmt["json_schema"]["strict"])

    @patch("fallback._groq", return_value={**VERDICT, "url": "https://example.net/invented"})
    @patch("fallback._post", return_value=EVIDENCE)
    def test_url_from_search_not_model(self, post, groq_):
        result = check_with_groq_tavily(CLAIM, "example", "example")
        self.assertEqual(result["sources"], {"https://example.org/source": "Archive"})
        post.assert_called_once()
        groq_.assert_called_once()

    @patch("fallback._post", return_value=EVIDENCE)
    def test_invalid_id_does_not_assign_verdict(self, _):
        with patch("fallback._groq", return_value={**VERDICT, "passage_id": 999}) as judge:
            with self.assertRaisesRegex(CheckError, "could not tie"):
                check_with_groq_tavily(CLAIM, "example", "example")
        self.assertEqual(judge.call_count, 2)

    @patch("fallback._post", return_value=EVIDENCE)
    def test_retries_invalid_id_and_quotes_exact_passage(self, post):
        with patch("fallback._groq", side_effect=[{**VERDICT, "passage_id": 99}, VERDICT]) as judge:
            result = check_with_groq_tavily(CLAIM, "example", "example")
        self.assertEqual(result["verdict"], "Verified")
        self.assertEqual(result["evidence_excerpt"], EVIDENCE["results"][0]["content"])
        self.assertEqual(result["sources"], {"https://example.org/source": "Archive"})
        self.assertEqual(judge.call_count, 2)
        self.assertNotIn("url", judge.call_args.args[2]["passages"][0])
        post.assert_called_once()

    @patch("fallback._groq", return_value=VERDICT)
    def test_unrelated_passage_is_not_a_verdict(self, _):
        wrong = {"url": "https://example.org/unrelated", "title": "Unrelated",
                 "content": "A plastic spoon was placed next to the unattended desk."}
        with patch("fallback._post", return_value={"results": [wrong]}):
            with self.assertRaisesRegex(CheckError, "could not tie"):
                check_with_groq_tavily(CLAIM, "example", "example")

    def test_second_extraction_recovers_non_numeric_status(self):
        missed = {"page": 1, "quote": "NASA has yet to bring a sample of asteroid Bennu back to Earth.",
                  "claim": "NASA has not yet returned a Bennu sample to Earth."}
        with patch("fallback._groq", side_effect=[{"claims": [CLAIM]}, {"claims": [missed]}]) as judge:
            got = extract_with_groq({1: CLAIM["quote"] + " " + missed["quote"]}, "example")
        self.assertEqual(got, [CLAIM, missed])
        self.assertEqual(judge.call_count, 2)
        self.assertNotIn(CLAIM["quote"], judge.call_args.args[2][1])

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

    def test_sdk_403_classifies_only_known_codes(self):
        for code, expected in [
            ("model_permission_blocked_project", "Groq project settings block the model"),
            ("model_permission_blocked_org", "Groq organization settings block the model"),
            ("private_unknown_code", "Groq denied this API request (403)"),
        ]:
            with self.subTest(code=code):
                def handler(request):
                    self.assertEqual(request.headers["Authorization"], "Bearer fake-offline-secret")
                    return httpx.Response(403, json={"error": {"code": code, "message": "private upstream body"}})

                client = Groq(api_key="fake-offline-secret", max_retries=0,
                              http_client=httpx.Client(transport=httpx.MockTransport(handler)))
                with patch("fallback.Groq", return_value=client):
                    with self.assertRaises(CheckError) as fault:
                        extract_with_groq({1: CLAIM["quote"]}, "fake-offline-secret")
                self.assertIn(expected, str(fault.exception))
                self.assertNotIn("private", str(fault.exception))
                self.assertNotIn("secret", str(fault.exception))
                self.assertTrue(fault.exception.__suppress_context__)


if __name__ == "__main__":
    unittest.main()
