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
    Returns:
      (A) EXACTLY 5 top-level bullets (1–5) with blank lines between bullets, and
      (B) a markdown table of the consolidated income statement (line-by-line) with %YoY and %QoQ for each item.
    Adds EBITDA and PAT margin %, and bps deltas YoY & QoQ with explicit QoQ/YoY definitions.
    """
    import json  # stdlib only

    f = _upload_to_openai(pdf_bytes, fname=f"{_slug(company or 'doc')}.pdf")

    # JSON prompt for higher determinism & clarity
    task = {
  "role": "equity_analyst",
  "objective": f"Read the attached BSE financial results PDF for {company or 'NA'} and produce a 5-bullet summary plus dated tables for P&L (and, if disclosed: Balance Sheet & Cash Flow). Prefer CONSOLIDATED; if unavailable, fall back to STANDALONE with clear labels.",
  "strict_scope": [
    "Prefer CONSOLIDATED statements (P&L, Balance Sheet, Cash Flow).",
    "If CONSOLIDATED is NOT available for a section, use the STANDALONE version for that section and clearly label it as 'Standalone'.",
    "If CONSOLIDATED is unavailable across ALL three statements, report ALL figures from STANDALONE and label every section and bullet as 'Standalone'.",
    "Normalize all monetary values to INR crore with 2 decimals. If a figure is unavailable, write 'Not disclosed'."
  ],
  "definitions_and_formulas": {
    "periods": {
      "quarter": "Assume the latest reported quarter unless explicitly stated otherwise.",
      "labeling_rules": [
        "Derive explicit period labels from the PDF (e.g., 'Quarter ended 30-Jun-2025' or 'Q1 FY26').",
        "Define placeholders you MUST replace in tables before printing: <Latest_qtr_label>, <Prev_qtr_label>, <YoY_qtr_label>.",
        "Latest_qtr = most recent reported quarter.",
        "Prev_qtr = immediately preceding quarter (QoQ comparator).",
        "YoY_qtr = same quarter in the prior fiscal year (YoY comparator)."
      ]
    },
    "metrics": {
      "Revenue": "As reported: 'Revenue from operations' (use 'Total income' only if 'Revenue from operations' is unavailable).",
      "EBITDA": "Revenue − (Total expenses before Depreciation/Amortisation and Finance Costs).",
      "PAT": "Profit after tax, as reported.",
      "EBITDA_margin_pct": "EBITDA / Revenue × 100.",
      "PAT_margin_pct": "PAT / Revenue × 100."
    },
    "line_item_mapping": {
      "Cost of Materials": ["Cost of materials consumed", "Raw materials consumed", "Raw material consumption"],
      "Employee Benefits": ["Employee benefits expense", "Staff costs", "Personnel expenses"],
      "Other expenses": ["Other expenses", "Operating and other expenses", "Miscellaneous expenses"]
    },
    "basis_selection": [
      "Set <scope_label> = 'Consolidated' when the relevant consolidated statement is found; else set <scope_label> = 'Standalone' for that section.",
      "Use the same basis for current, QoQ, and YoY comparators within a section to ensure % changes are consistent.",
      "If mixing bases across sections (e.g., P&L consolidated, Balance Sheet standalone), clearly label each section title with its basis."
    ],
    "change_calculations": {
      "QoQ": "Change for the latest quarter vs the immediately preceding quarter (difference in 3 months). Use percentage change for values. For margins, bps_change_qoq = (current_margin_pct − prior_quarter_margin_pct) × 100.",
      "YoY": "Change for the latest quarter vs the same quarter of the previous year (difference in 12 months). Use percentage change for values. For margins, bps_change_yoy = (current_margin_pct − prior_year_same_quarter_margin_pct) × 100."
    },
    "sign_conventions": "Use minus sign for negatives (e.g., -81.70 Cr, -153.2%). Round: values=2 decimals; percentages=1 decimal unless obvious; bps deltas as whole integers with 'bps'."
  },
  "output_contract": {
    "format": "plain_text",
    "sections": [
      {
        "name": "bullets_summary",
        "rules": [
          "Return EXACTLY FIVE top-level bullets, numbered 1–5.",
          "Each bullet MUST start at the beginning of a new line.",
          "Insert ONE blank line between bullets.",
          "Do not place bullets 2 or 3 on the same line as any other bullet."
        ],
        "templates": [
          "1- <scope_label> revenue stands at INR <Revenue Cr> (<YoY% YoY> / <QoQ% QoQ>)",

          "2- EBITDA stands at INR <EBITDA Cr> (<YoY% YoY>). EBITDA margin <expanded/contracted/changed> to <Margin %> (<±bps YoY> / <±bps QoQ>)",

          "3- Operating expenses snapshot:\\n   - Cost of Materials stands at INR <value Cr> (<YoY% YoY> / <QoQ% QoQ>)\\n   - Employee Benefits stands at INR <value Cr> (<YoY% YoY> / <QoQ% QoQ>)\\n   - Other expenses stands at INR <value Cr> (<YoY% YoY> / <QoQ% QoQ>)",

          "4- Finance Costs were a <key positive/drag>, <rising/declining> <YoY%> YoY to INR <value Cr>",

          "5- Net Profit/Loss after tax stands at INR <PAT Cr> (<YoY% YoY> / <QoQ% QoQ>). PAT margin <expanded/contracted/changed> to <Margin %> (<±bps YoY> / <±bps QoQ>)"
        ],
        "wording_rules": [
          "Use 'expanded' if the YoY bps change > 0, 'contracted' if < 0, else 'changed'.",
          "If a comparator is not available, write 'Not disclosed' for that comparator and omit the corresponding bps portion for margins.",
          "If any of the three expense line items are not disclosed, write 'Not disclosed' for its value and/or comparator(s)."
        ]
      },

      {
        "name": "consolidated_pnl_table",
        "title": "<scope_label> Income Statement (Quarter)",
        "render_as": "markdown_table",
        "columns": [
          "Line item",
          "<Latest_qtr_label> (INR Cr)",
          "<Prev_qtr_label> (INR Cr)",
          "<YoY_qtr_label> (INR Cr)",
          "%YoY",
          "%QoQ"
        ],
        "rows_inclusion_rules": [
          "Include line items as reported by the company; map common synonyms.",
          "Typical items (include if disclosed): Revenue from operations; Other income; Total income; Cost of materials consumed; Purchases of stock-in-trade; Changes in inventories of finished goods/work-in-progress/stock-in-trade; Employee benefits expense; Other expenses; EBITDA (computed); Finance costs; Depreciation and amortisation expense; Exceptional items (if any); Profit before tax (PBT); Tax expense; Profit/Loss after tax (PAT).",
          "For 'Cost of Materials' in the bullet summary, if the company reports 'Cost of materials consumed' and 'Purchases of stock-in-trade' separately, use 'Cost of materials consumed' alone for the bullet; do NOT sum unless the company explicitly provides a combined figure."
        ],
        "calc_rules": [
          "%YoY = ((Current − Same_qtr_last_year) / |Same_qtr_last_year|) × 100",
          "%QoQ = ((Current − Previous_qtr) / |Previous_qtr|) × 100",
          "Comparators must come from the same basis (<scope_label>) as the current period.",
          "For unavailable comparators or dates, print 'Not disclosed'."
        ]
      },

      {
        "name": "consolidated_balance_sheet_table",
        "title": "<scope_label> Balance Sheet (Statement of Assets & Liabilities)",
        "render_as": "markdown_table",
        "disclosure_rule": "If NEITHER consolidated nor standalone Balance Sheet is disclosed, output a single line: 'Balance Sheet: Not disclosed.'",
        "columns": [
          "Line item",
          "<As_of_latest_label> (INR Cr)",
          "<As_of_prev_qtr_label> (INR Cr)",
          "<As_of_yoy_qtr_label> (INR Cr)"
        ],
        "labeling_rules": [
          "As_of_latest_label = the 'as at' date corresponding to <Latest_qtr_label>.",
          "As_of_prev_qtr_label = the 'as at' date corresponding to <Prev_qtr_label>.",
          "As_of_yoy_qtr_label = the 'as at' date corresponding to <YoY_qtr_label>."
        ],
        "rows_inclusion_rules": [
          "Typical items (include if disclosed): Equity share capital; Other equity; Total equity; Borrowings (non-current); Borrowings (current); Lease liabilities; Deferred tax liabilities (net); Trade payables (current); Other financial liabilities; Provisions; Property, plant and equipment; Capital work-in-progress; Right-of-use assets; Goodwill/Intangibles; Inventories; Trade receivables; Cash and cash equivalents; Bank balances; Other current assets; Total assets; Total liabilities; Net debt (if reported)."
        ]
      },

      {
        "name": "consolidated_cashflow_table",
        "title": "<scope_label> Cash Flow Statement",
        "render_as": "markdown_table",
        "disclosure_rule": "If NEITHER consolidated nor standalone Cash Flow is disclosed, output a single line: 'Cash Flow Statement: Not disclosed.'",
        "columns": [
          "Line item",
          "<Latest_period_label> (INR Cr)",
          "<Prev_qtr_period_label> (INR Cr)",
          "<YoY_period_label> (INR Cr)"
        ],
        "labeling_notes": [
          "Respect the reporting basis: if the company provides quarter-only cash flows, use the quarter labels.",
          "If it provides YTD/H1/H9M basis, use those period labels aligned to the same cut-off dates as <Latest_qtr_label>, <Prev_qtr_label>, and <YoY_qtr_label>.",
          "If any comparator period is not provided, print 'Not disclosed'.",
          "Comparators must come from the same basis (<scope_label>) as the current period."
        ],
        "rows_inclusion_rules": [
          "Typical items (include if disclosed): Net cash from operating activities (CFO); Net cash used in investing activities (CFI); Net cash from/(used in) financing activities (CFF); Net increase/(decrease) in cash and cash equivalents; Cash and cash equivalents at end of period."
        ]
      }
    ]
  },
  "guardrails": [
    "Prefer CONSOLIDATED. If a section is missing in consolidated but available in standalone, use STANDALONE for that section and set <scope_label> accordingly.",
    "If CONSOLIDATED is entirely unavailable across P&L, Balance Sheet, and Cash Flow, report ALL data using STANDALONE and set <scope_label> = 'Standalone' globally.",
    "Keep units consistent (INR Cr).",
    "Do not invent figures; if a number cannot be found, write 'Not disclosed'.",
    "Before rendering tables and bullets, replace <scope_label>, <Latest_qtr_label>, <Prev_qtr_label>, <YoY_qtr_label> (and related 'as of'/'period' labels) with explicit basis and dates parsed from the PDF."
  ]
}



    task_json = json.dumps(task, ensure_ascii=False, indent=2)

    resp = client.responses.create(
        model=model,
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": task_json},
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
