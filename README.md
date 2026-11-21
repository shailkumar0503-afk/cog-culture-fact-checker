# Claim Check

A take-home starter that checks claims from a text PDF against live web sources. The interface and backend run in one Python Streamlit process. There is no frontend framework, database, or separate server to configure.

## Run this locally

1. Use Python 3.11 or 3.12. In this folder, run `python -m pip install -r requirements.txt`.
2. Create `.streamlit/secrets.toml` (the file is ignored by Git) and put this in it, substituting your own project key: `OPENAI_API_KEY = "your-key-here"`. Alternatively set the `OPENAI_API_KEY` environment variable. Use a project key with API web search access and active billing or credits. Do not put the key in the source, public repo, or screenshots.
3. Run `python -m streamlit run app.py`. Open the local URL shown.

### Optional provider: Groq + Tavily

Instead of setting an OpenAI key, set `GROQ_API_KEY` and `TAVILY_API_KEY` in your local or hosting secrets. Groq extracts and judges claims; Tavily performs the live searches. Set both keys, not just one. If both setups are configured, the app uses OpenAI and does not switch services mid-review. The UI identifies the selected providers before upload. The fallback supports up to 14,000 extracted characters per PDF and makes one basic Tavily lookup per claim. Free plans and provider access can change, so check the [Groq limits](https://console.groq.com/docs/rate-limits) and [Tavily pricing](https://docs.tavily.com/documentation/api-credits) before relying on them.

## Deploy and check before submitting

1. Push these files to the intended GitHub repository, leaving the secrets file out.
2. On [Streamlit Community Cloud](https://share.streamlit.io), deploy that repo with entry file `app.py`. In Advanced settings, set your provider key(s) under Secrets. Confirm the app is public and check the link in a private/incognito window. Deploying a private repository defaults to a private app; confirm the app permissions separately.
3. Upload `sample-test.pdf` (or create a fresh PDF) and compare against the expected checks below. Click the evidence links and verify that each source supports the actual wording and metric. Also try a blank/scan or a damaged PDF; those should get an input error, not a fact verdict.
4. Record a 30-second demo: upload, show the verdicts and correction, then click a source. Keep it as a backup if credits are exhausted. Confirm your final email includes the live app URL, GitHub repo, and GEO PowerPoint.

## Smoke test (synthetic PDF provided)

| Claim | Expected |
|---|---|
| The James Webb Space Telescope launched on December 25, 2021 (p. 1). | Verified; [NASA Webb launch timeline](https://science.nasa.gov/mission/webb/launch/). |
| NASA has yet to bring a sample of asteroid Bennu back to Earth (p. 1). | Inaccurate; [NASA reports the sample returned September 24, 2023](https://www.nasa.gov/news-release/nasas-first-asteroid-sample-has-landed-now-secure-in-clean-room/). |
| The Nancy Grace Roman Space Telescope team uses the internal name Project Saffron for its prelaunch checklist (p. 2). | False: no supporting evidence found; no fabricated correction. This claim is synthetic. |

`python -m unittest -v` runs offline tests using stub API responses. An API test and deployment are separate and require a real key and a public host.

## How it works

1. `pypdf` extracts selectable text, retaining each page number. An API call proposes discrete factual claims; local code checks that each reported quote appears on its claimed page.
2. Each claim gets a separate live web search using the OpenAI Responses API and GPT-5.4 mini, or Tavily search paired with Groq. The prompt compares scope, date, units, and sources rather than asking the model to recall a fact. Historical figures are not wrong just because a newer figure exists.
3. Results show the passage, page, check date, verdict and source links. A verified or inaccurate verdict is not shown without a source link from OpenAI citation metadata or a matching Tavily search excerpt. API errors are not verdicts.

## Troubleshooting and demo limits

- **App still asks for keys:** Check the deployed app's Secrets, including the exact key names. Do not commit a secrets file.
- **401 / 429 / service errors:** Confirm the selected providers’ keys, model or tool access, remaining quota, and (for OpenAI) API billing. Errors are not evidence and are never reported as false claims.
- **Sleep page:** [Streamlit Community Cloud](https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app#app-hibernation) hibernates apps after 12 hours without traffic. Wake and retest right before sending the link and note the wake button in your submission. If uninterrupted availability matters, use an already-authorized always-on host.
- **Keeping spend and latency bounded:** The demo limits uploads, checks at most 8 claims per review, and allows 2 concurrent reviews and 20 review starts per rolling day per process. Set `DEMO_DAILY_RUNS` in secrets to change the daily allowance (1–100). OpenAI checks request no more than 4 built-in calls per claim. These are demo controls, not billing caps: quotas reset on restart and are not shared across replicas. [OpenAI pricing](https://developers.openai.com/api/docs/pricing) bills both tool calls and tokens; use a dedicated project key, review provider-side limits, and set cost alerts before sharing.

## Scope and trade-offs

Text PDFs only, not scans or charts; a 5 MB, 12-page and 8-claim cap keeps a public demo bounded. Source presence is not proof of citation accuracy; an editor still needs to click and read it. No supporting evidence is not proof of falsity (the brief calls that class “False”). Do not upload confidential documents: their extracted text is processed by the selected model provider and a searchable claim may be sent to web search. Per-upload limits do not amount to billing limits; configure alerts and share the public demo only as intended. This starter doesn't persist document data in a database or filesystem.

This is a limited demo, not infrastructure for processing untrusted or sensitive PDFs at production scale.
