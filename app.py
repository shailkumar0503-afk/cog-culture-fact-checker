"""Streamlit entry point. Run with: python -m streamlit run app.py"""
import hashlib
import os
from collections import Counter
from datetime import datetime, timezone

import streamlit as st

from checker import CheckError, MAX_CLAIMS, check_claim, extract_claims, pdf_pages
from demo_limit import DemoLimit


@st.cache_resource
def demo_quota(runs):
    return DemoLimit(daily_runs=runs)


def setting(name, default=None):
    environment = os.getenv(name)
    if environment:
        return environment
    try:
        return st.secrets.get(name, default)
    except FileNotFoundError:
        return default


def daily_limit():
    try:
        value = int(setting("DEMO_DAILY_RUNS", 20))
    except (ValueError, TypeError):
        return 20
    return min(max(value, 1), 100)


def build_provider():
    """Use OpenAI when set; pair Groq with Tavily only when both keys are set."""
    openai_key = setting("OPENAI_API_KEY")
    if openai_key:
        return (
            "OpenAI web search",
            "The app sends extracted text to the OpenAI API; web search may receive the extracted claims.",
            lambda pages: extract_claims(pages, openai_key),
            lambda claim: check_claim(claim, openai_key),
        )
    groq_key, tavily_key = setting("GROQ_API_KEY"), setting("TAVILY_API_KEY")
    if groq_key and tavily_key:
        from fallback import check_with_groq_tavily, extract_with_groq
        return (
            "Groq + Tavily",
            "The app sends extracted text and retrieved evidence to Groq; extracted claims are used in Tavily web search queries.",
            lambda pages: extract_with_groq(pages, groq_key),
            lambda claim: check_with_groq_tavily(claim, groq_key, tavily_key),
        )
    return None


def run_review(data, extract, verify):
    pages = pdf_pages(data)
    quota = demo_quota(daily_limit())
    blocked = quota.enter()
    if blocked:
        raise CheckError(blocked)
    try:
        progress = st.progress(0, text="Reading the PDF and extracting factual claims…")
        try:
            claims = extract(pages)
            rows = []
            for index, claim in enumerate(claims, 1):
                progress.progress((index - 1) / len(claims),
                                  text=f"Searching the web for claim {index} of {len(claims)}…")
                try:
                    result = verify(claim)
                except CheckError as error:
                    result = {"verdict": "Not checked", "explanation": str(error), "sources": {}}
                rows.append({**claim, **result})
            return rows
        finally:
            progress.empty()
    finally:
        quota.leave()


def render_review(review):
    rows = review["rows"]
    if not rows:
        st.info("No specific, checkable claims were found in the selectable text.")
        return
    st.header("Review results")
    st.caption(f"{len(rows)} extracted claim(s) · Checked {review['date']} UTC · {review['provider']}")
    verdicts = ["Verified", "Inaccurate", "False", "Not checked"]
    counts = Counter(item["verdict"] for item in rows)
    for column, verdict in zip(st.columns(4), verdicts):
        column.metric("No support" if verdict == "False" else verdict, counts[verdict])
    labels = {"Verified": "🟢 Verified", "Inaccurate": "🟠 Inaccurate",
              "False": "⚪ False · No supporting evidence", "Not checked": "⚠ Not checked"}
    for index, row in enumerate(rows, 1):
        with st.container(border=True):
            st.caption(f"CLAIM {index:02d}  /  PAGE {row['page']}")
            st.subheader(labels[row["verdict"]])
            st.text(row["quote"])
            if row.get("correction"):
                st.markdown("**Corrected fact**")
                st.text(row["correction"])
            elif row["verdict"] == "Inaccurate":
                st.caption("This source indicates a mismatch, but no replacement was substantiated.")
            st.markdown("**Evidence**" if row["sources"] else "**Why this result**")
            st.text(row["explanation"])
            for url, title in row["sources"].items():
                st.link_button(title, url)


st.set_page_config(page_title="Claim Check", page_icon="🔎")
st.caption("EDITORIAL QA  /  LIVE WEB VERIFICATION")
st.title("Claim Check")
st.write("Check the factual claims in a PDF before you publish.")
provider = build_provider()
with st.container(border=True):
    upload = st.file_uploader("Upload a PDF", type=["pdf"], max_upload_size=5,
                              help=f"Select a text PDF with up to 12 pages and 5 MB. The app checks up to {MAX_CLAIMS} claims.")
    st.caption(f"Use public or synthetic documents only. {provider[1] if provider else 'When configured, this app uses external services to extract and verify claims.'}")
    run = st.button("Check claims", type="primary", disabled=upload is None)
with st.expander("How to read the verdicts"):
    st.write("Verified: evidence matches the claim. Inaccurate: evidence shows a mismatch, such as an outdated statistic. False: no reliable support found; that is not proof of falsehood. Not checked: a search or service problem prevented a verdict.")
data = upload.getvalue() if upload else None
digest = hashlib.sha256(data).hexdigest() if data else None
if st.session_state.get("review", {}).get("digest") != digest:
    st.session_state.pop("review", None)
if run and data:
    st.session_state.pop("review", None)
    if not provider:
        st.error("The app owner needs to configure an API key (OpenAI) or both fallback keys (Groq and Tavily). No check was run.")
    else:
        try:
            rows = run_review(data, provider[2], provider[3])
        except CheckError as error:
            st.error(str(error))
        else:
            st.session_state["review"] = {"digest": digest, "rows": rows, "provider": provider[0],
                                          "date": datetime.now(timezone.utc).date().isoformat()}
if "review" in st.session_state:
    render_review(st.session_state["review"])
