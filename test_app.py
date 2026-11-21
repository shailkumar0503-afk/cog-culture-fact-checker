import os
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from checker import CheckError
from demo_limit import DemoLimit
from test_checker import CLAIM, CITATION, FIXTURE


class DemoLimitTests(unittest.TestCase):
    def test_rolling_limit_and_concurrency(self):
        now = [0]
        gate = DemoLimit(daily_runs=2, active_runs=1, clock=lambda: now[0])
        self.assertEqual(gate.enter(), "")
        self.assertIn("already running", gate.enter())
        gate.leave()
        self.assertEqual(gate.enter(), "")
        gate.leave()
        self.assertIn("rolling", gate.enter())
        now[0] = 86400
        self.assertEqual(gate.enter(), "")
        gate.leave()


class AppTests(unittest.TestCase):
    def app(self):
        return AppTest.from_file(str(Path(__file__).with_name("app.py"))).run()

    def upload(self, app, data=None):
        app.get("file_uploader")[0].upload("sample-test.pdf", data or FIXTURE.read_bytes(),
                                            "application/pdf").run()

    def test_multiple_claims_corrections_service_failure_and_clear(self):
        app = self.app()
        self.assertFalse(app.exception)
        self.assertTrue(app.button[0].proto.disabled)
        self.upload(app)
        delayed_claims = [CLAIM, {**CLAIM, "page": 1, "quote": "No Bennu sample has returned."},
                         {**CLAIM, "page": 2, "quote": "The unsupported claim."},
                         {**CLAIM, "page": 2, "quote": "Test service failure."}]
        cite = {CITATION['url']: CITATION['title']}
        responses = [
            {"verdict": "Verified", "explanation": "Matches official announcement.", "sources": cite},
            {"verdict": "Inaccurate", "explanation": "NASA reported the sample landing.",
             "correction": "NASA returned a Bennu sample in 2023.", "sources": cite},
            {"verdict": "False", "explanation": "No supporting evidence found.", "sources": {}},
            CheckError("API is unavailable."),
        ]
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}), \
             patch("checker.extract_claims", return_value=delayed_claims), \
             patch("checker.check_claim", side_effect=responses):
            app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual([item.value for item in app.metric], ["1"] * 4)
        self.assertEqual(len(app.subheader), 4)
        self.assertIn("NASA returned a Bennu sample in 2023.", [item.value for item in app.text])
        self.assertIn("API is unavailable.", [item.value for item in app.text])
        self.assertEqual(len(app.get("link_button")), 2)
        app.run()
        self.assertEqual(len(app.subheader), 4)
        app.get("file_uploader")[0].clear().run()
        self.assertTrue(app.button[0].proto.disabled)
        self.assertEqual(len(app.subheader), 0)

    def test_missing_key_and_invalid_pdf(self):
        app = self.app()
        self.upload(app)
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            app.button[0].click().run()
        self.assertIn("configure an API key", app.error[0].value)
        self.upload(app, b"not a pdf")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test"}), \
             patch("checker.extract_claims") as extract:
            app.button[0].click().run()
        extract.assert_not_called()
        self.assertIn("valid PDF", app.error[0].value)

    def test_fallback_selection_and_provider_notice(self):
        app = self.app()
        self.upload(app)
        creds = {"OPENAI_API_KEY": "", "GROQ_API_KEY": "groq-local-test",
                 "TAVILY_API_KEY": "tavily-local-test"}
        result = {"verdict": "False", "explanation": "No reliable support.", "sources": {}}
        with patch.dict(os.environ, creds), \
             patch("fallback.extract_with_groq", return_value=[CLAIM]) as extract, \
             patch("fallback.check_with_groq_tavily", return_value=result) as verify:
            app.button[0].click().run()
            verify.assert_called_once_with(CLAIM, creds["GROQ_API_KEY"], creds["TAVILY_API_KEY"])
            self.assertEqual(extract.call_args[0][1], creds["GROQ_API_KEY"])
        self.assertFalse(app.exception)
        captions = " ".join(caption.value for caption in app.caption)
        self.assertIn("Groq + Tavily", captions)
        self.assertIn("Tavily web search queries", captions)
        self.assertNotIn(creds["GROQ_API_KEY"], captions)
        with patch.dict(os.environ, {**creds, "OPENAI_API_KEY": "openai-local-test"}), \
             patch("checker.extract_claims", return_value=[CLAIM]) as openai_extract, \
             patch("checker.check_claim", return_value=result), \
             patch("fallback.extract_with_groq") as groq_extract:
            app.button[0].click().run()
            openai_extract.assert_called_once()
            groq_extract.assert_not_called()
        self.assertIn("OpenAI web search", " ".join(caption.value for caption in app.caption))
