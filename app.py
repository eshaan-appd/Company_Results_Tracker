import os, io, re, time, tempfile
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import pandas as pd
import numpy as np
from openai import OpenAI
import os, streamlit as st
# ---- OpenAI (Responses API) ----

api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
if not api_key:
    st.error("Missing OPENAI_API_KEY (set env var or add to Streamlit secrets).")
    st.stop()

client = OpenAI(api_key=api_key)

with st.expander("🔍 OpenAI connection diagnostics", expanded=False):
    # 1) Is the key visible?
    key_src = "st.secrets" if "OPENAI_API_KEY" in st.secrets else "env"
    mask = lambda s: (s[:7] + "..." + s[-4:]) if s and len(s) > 12 else "unset"
    st.write("Key source:", key_src)
    st.write("API key (masked):", mask(st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")))

    # 2) Can we list models? (auth + project/perm sanity)
    try:
        _ = client.models.list()
        st.success("Models list ok — auth + project look good.")
    except Exception as e:
        st.error(f"Models list failed: {e}")

    # 3) Can we call a tiny Responses echo? (billing/quota often shows up here)
    try:
        r = client.responses.create(model="gpt-4.1-mini", input="ping")
        st.success("Responses call ok.")
    except Exception as e:
        st.error(f"Responses call failed: {e}")

# =========================================
# Streamlit UI
# =========================================
st.set_page_config(page_title="Listed Company Results Tracker", layout="wide")
st.title("📈 Listed Company Results Tracker")
st.caption("Fetch Result Announcements → Pick PDF → Upload to OpenAI → Summarize the results")

# =========================================
# Small utilities
# =========================================
_ILLEGAL_RX = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
def _clean(s: str) -> str:
    return _ILLEGAL_RX.sub('', s) if isinstance(s, str) else s

def _first_col(df: pd.DataFrame, names):
    for n in names:
        if n in df.columns: return n
    return None

def _norm(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()

def _slug(s: str, maxlen: int = 60) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", str(s or "")).strip("_")
    return (s[:maxlen] if len(s) > maxlen else s) or "file"

# =========================================
# Attachment URL candidates (unchanged)
# =========================================
def _candidate_urls(row):
    cands = []
    att = str(row.get("ATTACHMENTNAME") or "").strip()
    if att:
        cands += [
            f"https://www.bseindia.com/xml-data/corpfiling/AttachHis/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/Attach/{att}",
            f"https://www.bseindia.com/xml-data/corpfiling/AttachLive/{att}",
        ]
    ns = str(row.get("NSURL") or "").strip()
    if ".pdf" in ns.lower():
        cands.append(ns if ns.lower().startswith("http") else "https://www.bseindia.com/" + ns.lstrip("/"))
    seen, out = set(), []
    for u in cands:
        if u and u not in seen:
            out.append(u); seen.add(u)
    return out

# =========================================
# BSE fetch — strict; returns filtered DF (unchanged logic)
# =========================================
def fetch_bse_announcements_strict(start_yyyymmdd: str,
                                   end_yyyymmdd: str,
                                   verbose: bool = True,
                                   request_timeout: int = 25) -> pd.DataFrame:
    """Fetches raw announcements, then filters:
    Category='Company Update' AND subcategory contains any:
    Acquisition | Amalgamation / Merger | Scheme of Arrangement | Joint Venture
    """
    assert len(start_yyyymmdd) == 8 and len(end_yyyymmdd) == 8
    assert start_yyyymmdd <= end_yyyymmdd
    base_page = "https://www.bseindia.com/corporates/ann.html"
    url = "https://api.bseindia.com/BseIndiaAPI/api/AnnSubCategoryGetData/w"

    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": base_page,
        "X-Requested-With": "XMLHttpRequest",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    })

    try:
        s.get(base_page, timeout=15)
    except Exception:
        pass

    variants = [
        {"subcategory": "-1", "strSearch": "P"},
        {"subcategory": "-1", "strSearch": ""},
        {"subcategory": "",   "strSearch": "P"},
        {"subcategory": "",   "strSearch": ""},
    ]

    all_rows = []
    for v in variants:
        params = {
            "pageno": 1, "strCat": "-1", "subcategory": v["subcategory"],
            "strPrevDate": start_yyyymmdd, "strToDate": end_yyyymmdd,
            "strSearch": v["strSearch"], "strscrip": "", "strType": "C",
        }
        rows, total, page = [], None, 1
        while True:
            r = s.get(url, params=params, timeout=request_timeout)
            ct = r.headers.get("content-type","")
            if "application/json" not in ct:
                if verbose: st.warning(f"[variant {v}] non-JSON on page {page} (ct={ct}).")
                break
            data = r.json()
            table = data.get("Table") or []
            rows.extend(table)
            if total is None:
                try: total = int((data.get("Table1") or [{}])[0].get("ROWCNT") or 0)
                except Exception: total = None
            if not table: break
            params["pageno"] += 1; page += 1; time.sleep(0.25)
            if total and len(rows) >= total: break
        if rows:
            all_rows = rows; break

    if not all_rows: return pd.DataFrame()

    all_keys = set()
    for r in all_rows: all_keys.update(r.keys())
    df = pd.DataFrame(all_rows, columns=list(all_keys))

    # Filter to Result
    def filter_announcements(df_in: pd.DataFrame, category_filter="Result") -> pd.DataFrame:
        if df_in.empty: return df_in.copy()
        cat_col = _first_col(df_in, ["CATEGORYNAME","CATEGORY","NEWS_CAT","NEWSCATEGORY","NEWS_CATEGORY"])
        if not cat_col: return df_in.copy()
        df2 = df_in.copy()
        df2["_cat_norm"] = df2[cat_col].map(lambda x: _norm(x).lower())
        return df2.loc[df2["_cat_norm"] == _norm(category_filter).lower()].drop(columns=["_cat_norm"])

    df_filtered = filter_announcements(df, category_filter="Result")

    return df_filtered

# =========================================
# OpenAI PDF summarization
# =========================================
def _download_pdf(url: str, timeout=25) -> bytes:
    s = requests.Session()
    s.headers.update({
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": "https://www.bseindia.com/corporates/ann.html",
        "Accept-Language": "en-US,en;q=0.9",
    })
    r = s.get(url, timeout=timeout, allow_redirects=True, stream=False)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}")
    data = r.content
    if not (data[:8].startswith(b"%PDF") or url.lower().endswith(".pdf")):
        # some BSE PDFs have no content-type; head check is enough
        pass
    return data

def _upload_to_openai(pdf_bytes: bytes, fname: str = "document.pdf"):
    # The Files API stores the PDF so a model can read it; then we attach it in a Responses call.
    # Using a temp file ensures a valid filename is sent.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        tmp.flush()
        # purpose "assistants" is the general bucket for file inputs / tools
        f = client.files.create(file=open(tmp.name, "rb"), purpose="assistants")
    return f

# Keep your existing imports; do NOT add any new ones.

def _ensure_main_bullet_newlines(text: str) -> str:
    """
    Ensure bullets 1–5 each begin on a new line with a blank line
    before bullets 2–5. Uses existing 're' (already imported).
    """
    text = re.sub(r'^\s+', '', text or '')  # trim leading whitespace
    for n in range(2, 6):
        # If bullet 'n- ' is not preceded by a newline, insert two newlines
        text = re.sub(rf'(?<!\n)\s*{n}-\s', f'\n\n{n}- ', text)
    return text.strip()

def summarize_pdf_with_openai(
    pdf_bytes: bytes,
    company: str,
    headline: str,
    subcat: str,
    model: str = "gpt-4.1-mini",
    style: str = "bullets",
    max_output_tokens: int = 800,
    temperature: float = 0.2
) -> str:
    """
    Uses the Responses API with a file attachment.
    Returns a consolidated-results summary in a fixed 5-bullet format,
    with each bullet on its own line and a blank line between bullets.
    """
    f = _upload_to_openai(pdf_bytes, fname=f"{_slug(company or 'doc')}.pdf")

    task = f"""
You are an equity analyst. Read the attached BSE financial results PDF for {company or 'NA'}.

STRICT SCOPE
- Use the CONSOLIDATED statements only (P&L and Balance Sheet). Ignore standalone.
- Normalize to INR crore (2 decimals). If a figure is unavailable, write "Not disclosed".

CALCULATIONS
- EBITDA = Revenue − (Total expenses before Depreciation/Amortisation and Finance Costs).
- EBITDA margin (%) = EBITDA / Revenue × 100.
- YoY = vs same quarter last year (or same period for HY/FY).
- QoQ = vs immediately preceding quarter (for quarterly results only).

OUTPUT FORMAT — PLAIN TEXT ONLY
- Return EXACTLY FIVE top-level bullets, numbered 1–5.
- **Each bullet MUST start at the beginning of a new line, with ONE blank line between bullets.**
- **Do not place bullets 2 or 3 on the same line as any other bullet.**

Bullets to produce:

1- Consolidated revenue stands at INR <Revenue Cr> (<YoY% YoY> / <QoQ% QoQ>)
2- EBITDA stands at INR <EBITDA Cr> (<YoY% YoY>). EBITDA margin <expanded/contracted> to <Margin %> (<±bps> YoY)
3- Key expense lines like Cost of Materials and Employee Benefits grew faster/slower than revenue. Provide two facts:
   - Cost of Materials <rose/fell> <YoY%> YoY to INR <value Cr>
   - Employee Benefits <rose/fell> <YoY%> YoY to INR <value Cr>
4- Finance Costs were a <key positive/drag>, <rising/declining> <YoY%> YoY to INR <value Cr>
5- Net Profit/Loss after tax stands at INR <value Cr> (<YoY% YoY> / <QoQ% QoQ>)

Rules:
- Use minus sign for negatives (e.g., -81.70 Cr, -153.2%).
- If QoQ is not applicable, write "Not disclosed".
"""

    resp = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": task},
                {"type": "input_file", "file_id": f.id},
            ],
        }],
    )
    raw = (resp.output_text or "").strip()
    return _ensure_main_bullet_newlines(raw)



# Simple rate-limit friendly wrapper
def safe_summarize(*args, **kwargs) -> str:
    for i in range(4):
        try:
            return summarize_pdf_with_openai(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                time.sleep(2.0 * (i + 1))
                continue
            raise
    return "⚠️ Unable to summarize due to repeated rate limits."

# =========================================
# Sidebar controls
# =========================================
with st.sidebar:
    st.header("⚙️ Controls")
    today = datetime.now().date()
    start_date = st.date_input("Start date", value=today - timedelta(days=1), max_value=today)
    end_date   = st.date_input("End date", value=today, max_value=today, min_value=start_date)

    model = st.selectbox(
        "OpenAI model",
        ["gpt-4.1-mini", "gpt-4o-mini", "gpt-4.1"],
        index=0,
        help="Models with vision/file-reading capability. 4.1-mini/4o-mini are cost-efficient."
    )
    style = st.radio("Summary style", ["bullets", "narrative"], horizontal=True)
    max_tokens = st.slider("Max output tokens", 200, 2000, 800, step=50)
    temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, step=0.1)

    max_workers = st.slider("Parallel summaries", 1, 8, 3, help="Lower if you hit 429s.")
    max_items = st.slider("Max announcements to summarize", 5, 200, 60, step=5)

    run = st.button("🚀 Fetch & Summarize with OpenAI", type="primary")

# =========================================
# Run pipeline (fetch → PDFs → OpenAI summaries)
# =========================================
def _fmt(d: datetime.date) -> str: return d.strftime("%Y%m%d")

def _pick_company_cols(df: pd.DataFrame) -> tuple[str, str]:
    nm = _first_col(df, ["SLONGNAME","SNAME","SC_NAME","COMPANYNAME"]) or "SLONGNAME"
    subcol = _first_col(df, ["SUBCATEGORYNAME","SUBCATEGORY","SUB_CATEGORY","NEWS_SUBCATEGORY"]) or "SUBCATEGORYNAME"
    return nm, subcol

if run:
    if not os.getenv("OPENAI_API_KEY"):
        st.error("Missing OPENAI_API_KEY (set env var, add to Streamlit Secrets, or export in your shell).")
        st.stop()

    start_str, end_str = _fmt(start_date), _fmt(end_date)

    with st.status("Fetching announcements…", expanded=True):
        df_hits = fetch_bse_announcements_strict(start_str, end_str, verbose=False)
        st.write(f"Matched rows after filters: **{len(df_hits)}**")

    if df_hits.empty:
        st.warning("No matching announcements in this window.")
        st.stop()

    if len(df_hits) > max_items:
        df_hits = df_hits.head(max_items)

    # Build list of PDF targets
    rows = []
    for _, r in df_hits.iterrows():
        urls = _candidate_urls(r)
        rows.append((r, urls))

    st.subheader("📑 Summaries (OpenAI)")
    nm, subcol = _pick_company_cols(df_hits)

    # Worker to download and summarize a single row
    def worker(idx, row, urls):
        # try urls in order until one downloads
        pdf_bytes, used_url = None, ""
        for u in urls:
            try:
                data = _download_pdf(u, timeout=25)
                if data and len(data) > 500:
                    pdf_bytes, used_url = data, u
                    break
            except Exception:
                continue
        if not pdf_bytes:
            return idx, used_url, "⚠️ Could not fetch a valid PDF.", None

        company = str(row.get(nm) or "").strip()
        headline = str(row.get("HEADLINE") or "").strip()
        subcat = str(row.get(subcol) or "").strip()

        summary = safe_summarize(pdf_bytes, company, headline, subcat,
                                 model=model,
                                 style=("bullets" if style=="bullets" else "narrative"),
                                 max_output_tokens=int(max_tokens),
                                 temperature=float(temperature))
        return idx, used_url, summary, None

    # Run with limited parallelism
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(worker, i, r, urls) for i, (r, urls) in enumerate(rows)]
        for fut in as_completed(futs):
            i, pdf_url, summary, _ = fut.result()
            r = rows[i][0]
            company = str(r.get(nm) or "").strip()
            dt = str(r.get("NEWS_DT") or "").strip()
            subcat = str(r.get(subcol) or "").strip()
            headline = str(r.get("HEADLINE") or "").strip()

            with st.expander(f"{company or 'Unknown'} — {dt}  •  {subcat or 'N/A'}", expanded=False):
                if headline:
                    st.markdown(f"**Headline:** {headline}")
                if pdf_url:
                    st.markdown(f"[PDF link]({pdf_url})")
                st.markdown(summary)

else:
    st.info("Pick your date range and click **Fetch & Summarize with OpenAI**. This version uploads each PDF to OpenAI and renders the model’s summary right here.")
