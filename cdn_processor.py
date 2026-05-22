"""
cdn_processor.py
================
Performs all heavy CDN analytics computations using Pandas.
Receives pre-aggregated SQL summaries from Laravel and returns
the same response shape the Vue dashboard already expects.

Key feature: PATH INTELLIGENCE ENGINE
  All URL-bearing DataFrames are normalized before scoring.
  Tracking parameters (UTM, gclid, fbclid, etc.) are stripped,
  trailing slashes are standardized, and duplicate raw URLs that
  canonicalize to the same path are merged with weighted averages.

Data contract (input keys):
  daily_summary   : [{date, total, ad_hits}]           — 30 rows
  page_stats      : [{page_url, total_hits, avg_duration,
                      avg_scroll, avg_clicks, ad_hits,
                      today_count, yesterday_count}]   — top N pages
  session_counts  : [{page_url, session_id, hit_count}] — for bounce rate
  sparkline_raw   : [{page_url, date, cnt}]            — 14-day daily
  velocity_raw    : [{page_url, last7, prev7}]
  geo_raw         : [{country_code, city, device_type, count}]
  referrers       : [{referrer, count}]
  errors          : [{url, error_type, load_time_ms, created_at}]
  keywords        : [{query, intent}]
  meta            : {today, yesterday, pages_page, pages_per_page,
                     normalize_ids}   ← optional bool, default False
"""

import re
import pandas as pd
import numpy as np
from urllib.parse import urlparse, parse_qs, urlencode
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)

def safe_int(val: Any, default: int = 0) -> int:
    """Safely converts a value to an integer, handling NaN and None."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return int(float(val))
    except (ValueError, TypeError, OverflowError):
        return default


# ─────────────────────────────────────────────────────────────────────────────
# PATH INTELLIGENCE ENGINE
# ─────────────────────────────────────────────────────────────────────────────

# Every query-string key in this set is pure tracking noise.
# It carries zero information about which content the user viewed.
_TRACKING_PARAMS: Set[str] = {
    # UTM (Google Analytics campaign tags)
    "utm_source", "utm_medium", "utm_campaign", "utm_content",
    "utm_term", "utm_id", "utm_reader",
    # Google Ads / DoubleClick
    "gclid", "gclsrc", "dclid", "_ga", "_gid", "_gl", "gbraid", "wbraid",
    # Meta / Facebook
    "fbclid", "fb_action_ids", "fb_action_types", "fb_source", "fb_ref",
    # Microsoft / Bing
    "msclkid",
    # HubSpot
    "hsa_cam", "hsa_grp", "hsa_mt", "hsa_src", "hsa_ad",
    "hsa_acc", "hsa_net", "hsa_kw", "hsa_tgt", "hsa_ver",
    # Mailchimp
    "mc_cid", "mc_eid",
    # Twitter / X
    "twclid",
    # TikTok
    "ttclid",
    # Zanox / Awin affiliate
    "zanpid", "awc",
    # LinkedIn
    "li_fat_id",
    # Generic noise
    "ref", "referrer", "source", "origin", "campaign",
    "rand", "random", "nocache", "preview", "cachebust",
    # Internal framework params often leaked into URLs
    "_", "v", "ver", "ts", "timestamp", "cb",
}

# Path-segment patterns that represent dynamic IDs, not content names.
_PURE_INT_RE  = re.compile(r"^\d{1,20}$")
_UUID_RE      = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)
# Slug ending in a long numeric suffix: "my-article-20491"
_SLUG_INT_RE  = re.compile(r"^[a-z0-9][a-z0-9-]*-\d{4,}$")


def normalize_url(raw_url: str, collapse_ids: bool = False) -> str:
    """
    Returns the canonical form of a URL:

    1. Lowercase scheme + host
    2. Strip all known tracking query parameters
    3. Keep meaningful query params in deterministic sorted order
    4. Remove URL fragment (#section)
    5. Strip trailing slash (except bare root "/")
    6. Optionally replace numeric/UUID path segments with {id}

    Examples
    --------
    normalize_url("https://Example.com/Blog/?utm_source=google&page=2")
    → "https://example.com/Blog?page=2"

    normalize_url("https://example.com/product/12345", collapse_ids=True)
    → "https://example.com/product/{id}"
    """
    if not raw_url or not isinstance(raw_url, str):
        return raw_url or ""

    try:
        p = urlparse(raw_url.strip())
    except Exception:
        return raw_url

    scheme = (p.scheme or "https").lower()
    host   = (p.netloc or "").lower()
    path   = p.path or "/"

    # Normalize trailing slash
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Optionally collapse dynamic path segments
    if collapse_ids:
        parts = path.split("/")
        path = "/".join(
            "{id}"
            if (_PURE_INT_RE.match(s) or _UUID_RE.match(s) or _SLUG_INT_RE.match(s))
            else s
            for s in parts
        )

    # Strip tracking params, keep meaningful ones in sorted order
    clean_query = ""
    if p.query:
        qs = parse_qs(p.query, keep_blank_values=False)
        meaningful = {
            k: v
            for k, v in qs.items()
            if k.lower() not in _TRACKING_PARAMS
        }
        if meaningful:
            clean_query = urlencode(sorted(meaningful.items()), doseq=True)

    canonical = f"{scheme}://{host}{path}"
    if clean_query:
        canonical += f"?{clean_query}"

    return canonical


def _apply_path_intelligence(
    page_df:    pd.DataFrame,
    session_df: pd.DataFrame,
    spark_df:   pd.DataFrame,
    vel_df:     pd.DataFrame,
    err_df:     pd.DataFrame,
    collapse_ids: bool = False,
) -> tuple:
    """
    Applies URL normalization to every DataFrame that carries a URL column,
    then merges rows that collapse to the same canonical URL.

    Returns the same 5 DataFrames in normalized form, plus a dict:
      {canonical_url → {url_variants: int, raw_urls: [str]}}
    that the caller can attach to top_pages for UI transparency.
    """

    # ── 1. Build canonical map for page_stats ────────────────────────────────
    variant_registry: Dict[str, Dict] = {}  # canonical → {raw_urls, count}

    if not page_df.empty and "page_url" in page_df.columns:
        page_df = page_df.copy()
        page_df["canonical_url"] = page_df["page_url"].apply(
            lambda u: normalize_url(u, collapse_ids)
        )

        # Register variants
        for _, row in page_df.iterrows():
            raw  = row["page_url"]
            canon = row["canonical_url"]
            if canon not in variant_registry:
                variant_registry[canon] = {"raw_urls": [], "url_variants": 0}
            if raw not in variant_registry[canon]["raw_urls"]:
                variant_registry[canon]["raw_urls"].append(raw)
                variant_registry[canon]["url_variants"] += 1

        # Merge rows that share the same canonical URL.
        # Hits → sum.  Averages → weighted mean by total_hits.
        numeric_cols = ["avg_duration", "avg_scroll", "avg_clicks"]
        page_df["total_hits"]     = pd.to_numeric(page_df["total_hits"],     errors="coerce").fillna(0)
        page_df["ad_hits"]        = pd.to_numeric(page_df["ad_hits"],        errors="coerce").fillna(0)
        page_df["today_count"]    = pd.to_numeric(page_df["today_count"],    errors="coerce").fillna(0)
        page_df["yesterday_count"]= pd.to_numeric(page_df["yesterday_count"],errors="coerce").fillna(0)

        for col in numeric_cols:
            page_df[col] = pd.to_numeric(page_df.get(col, 0), errors="coerce").fillna(0)
            # Weighted sum = value × weight; we'll divide by total_hits after groupby
            page_df[f"_{col}_weighted"] = page_df[col] * page_df["total_hits"]

        agg_rules: Dict[str, Any] = {
            "total_hits":       ("total_hits",      "sum"),
            "ad_hits":          ("ad_hits",         "sum"),
            "today_count":      ("today_count",     "sum"),
            "yesterday_count":  ("yesterday_count", "sum"),
            # Keep the first raw URL as the display label
            "display_url":         ("page_url",        "first"),
        }
        for col in numeric_cols:
            agg_rules[f"_{col}_wsum"]  = (f"_{col}_weighted", "sum")

        grouped = page_df.groupby("canonical_url", sort=False).agg(**agg_rules).reset_index()

        # Recover weighted averages
        for col in numeric_cols:
            grouped[col] = np.where(
                grouped["total_hits"] > 0,
                grouped[f"_{col}_wsum"] / grouped["total_hits"],
                0.0,
            )
            grouped.drop(columns=[f"_{col}_wsum"], inplace=True)

        page_df = grouped.rename(columns={"canonical_url": "page_url"})

    # ── 2. Normalize session_counts ──────────────────────────────────────────
    if not session_df.empty and "page_url" in session_df.columns:
        session_df = session_df.copy()
        session_df["page_url"] = session_df["page_url"].apply(
            lambda u: normalize_url(u, collapse_ids)
        )

    # ── 3. Normalize sparkline_raw ───────────────────────────────────────────
    if not spark_df.empty and "page_url" in spark_df.columns:
        spark_df = spark_df.copy()
        spark_df["page_url"] = spark_df["page_url"].apply(
            lambda u: normalize_url(u, collapse_ids)
        )
        # Merge rows with same canonical_url + date
        spark_df = (
            spark_df.groupby(["page_url", "date"], sort=False)["cnt"]
            .sum()
            .reset_index()
        )

    # ── 4. Normalize velocity_raw ────────────────────────────────────────────
    if not vel_df.empty and "page_url" in vel_df.columns:
        vel_df = vel_df.copy()
        vel_df["page_url"] = vel_df["page_url"].apply(
            lambda u: normalize_url(u, collapse_ids)
        )
        vel_df["last7"] = pd.to_numeric(vel_df["last7"], errors="coerce").fillna(0)
        vel_df["prev7"] = pd.to_numeric(vel_df["prev7"], errors="coerce").fillna(0)
        vel_df = vel_df.groupby("page_url", sort=False).agg(
            last7=("last7", "sum"),
            prev7=("prev7", "sum"),
        ).reset_index()

    # ── 5. Normalize error URLs ───────────────────────────────────────────────
    if not err_df.empty and "url" in err_df.columns:
        err_df = err_df.copy()
        err_df["url"] = err_df["url"].apply(lambda u: normalize_url(u, collapse_ids))

    return page_df, session_df, spark_df, vel_df, err_df, variant_registry




# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def analyze(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point called from the FastAPI endpoint.
    Orchestrates all sub-computations and returns the full analytics payload.
    """
    meta            = data.get("meta", {})
    today_str       = meta.get("today",     datetime.utcnow().strftime("%Y-%m-%d"))
    yesterday_str   = meta.get("yesterday", (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d"))
    collapse_ids    = bool(meta.get("normalize_ids", False))

    daily_summary   = data.get("daily_summary",  [])
    page_stats      = data.get("page_stats",     [])
    session_counts  = data.get("session_counts", [])
    sparkline_raw   = data.get("sparkline_raw",  [])
    velocity_raw    = data.get("velocity_raw",   [])
    geo_raw         = data.get("geo_raw",        [])
    referrers       = data.get("referrers",      [])
    errors          = data.get("errors",         [])
    keywords        = data.get("keywords",       [])
    pages_page      = int(meta.get("pages_page",     1))
    pages_per_page  = int(meta.get("pages_per_page", 10))
    pages_total     = int(meta.get("pages_total",    len(page_stats)))

    # ── Build raw DataFrames ──────────────────────────────────────────────────
    daily_df    = pd.DataFrame(daily_summary)   if daily_summary  else pd.DataFrame()
    page_df     = pd.DataFrame(page_stats)      if page_stats     else pd.DataFrame()
    session_df  = pd.DataFrame(session_counts)  if session_counts else pd.DataFrame()
    spark_df    = pd.DataFrame(sparkline_raw)   if sparkline_raw  else pd.DataFrame()
    vel_df      = pd.DataFrame(velocity_raw)    if velocity_raw   else pd.DataFrame()
    geo_df      = pd.DataFrame(geo_raw)         if geo_raw        else pd.DataFrame()
    ref_df      = pd.DataFrame(referrers)       if referrers      else pd.DataFrame()
    err_df      = pd.DataFrame(errors)          if errors         else pd.DataFrame()

    # ── PATH INTELLIGENCE — normalize & deduplicate all URL columns ──────────
    page_df, session_df, spark_df, vel_df, err_df, variant_registry = \
        _apply_path_intelligence(
            page_df, session_df, spark_df, vel_df, err_df,
            collapse_ids=collapse_ids,
        )

    logger.info(
        "Path Intelligence: %d raw pages → %d canonical paths (collapse_ids=%s)",
        len(page_stats), len(page_df), collapse_ids,
    )

    # ── Run computations ──────────────────────────────────────────────────────
    daily_history   = _build_daily_history(daily_df, today_str)
    summary         = _build_summary(daily_history, today_str, yesterday_str)
    bounce_by_page  = _compute_bounce_rates(session_df)
    errors_by_page  = _aggregate_errors_by_page(err_df)
    top_pages       = _build_top_pages(page_df, spark_df, bounce_by_page, errors_by_page, keywords, variant_registry, today_str=today_str)
    top_referrers   = _build_top_referrers(ref_df)
    trend_velocity  = _build_velocity(vel_df)
    by_country, by_city, by_device = _build_geo(geo_df)
    site_health     = _build_site_health(err_df)

    return {
        "daily_history":       daily_history,
        "top_pages":           top_pages,
        "pages_total":         pages_total,
        "pages_page":          pages_page,
        "pages_per_page":      pages_per_page,
        "top_referrers":       top_referrers,
        "trend_velocity":      trend_velocity,
        "by_country":          by_country,
        "by_device":           by_device,
        "by_city":             by_city,
        "site_health":         site_health,
        "summary":             summary,
        "path_intelligence":   {
            "raw_page_count":       len(page_stats),
            "canonical_page_count": len(page_df),
            "duplicates_merged":    max(0, len(page_stats) - len(page_df)),
            "collapse_ids_active":  collapse_ids,
        },
    }



# ─────────────────────────────────────────────────────────────────────────────
# DAILY HISTORY
# ─────────────────────────────────────────────────────────────────────────────

def _build_daily_history(df: pd.DataFrame, today_str: str) -> List[Dict]:
    """
    Returns 30 consecutive days with zero-fill for missing dates.
    Input df has columns: date, total, ad_hits.
    """
    def safe_int(val: Any) -> int:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    today = datetime.strptime(today_str, "%Y-%m-%d")

    # Build an index keyed by date string
    daily_history = []
    if not df.empty and "date" in df.columns:
        for row in df.to_dict("records"):
            daily_history.append({
                "date":    str(row.get("date", "")),
                "total":   safe_int(row.get("total")),
                "ad_hits": safe_int(row.get("ad_hits")),
            })


    history = []
    for i in range(29, -1, -1):
        day = today - timedelta(days=i)
        d   = day.strftime("%Y-%m-%d")
        
        match = next((item for item in daily_history if item["date"] == d), {})
        history.append({
            "date":    d,
            "label":   day.strftime("%b %-d"),
            "total":   match.get("total", 0),
            "ad_hits": match.get("ad_hits", 0),
        })
    return history


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def _build_summary(daily_history: List[Dict], today_str: str, yesterday_str: str) -> Dict:
    """Derives today vs yesterday and 7d vs prev-7d deltas."""
    by_date = {d["date"]: d["total"] for d in daily_history}

    today_hits     = by_date.get(today_str,     0)
    yesterday_hits = by_date.get(yesterday_str, 0)

    today_delta = None
    if yesterday_hits > 0:
        today_delta = round(((today_hits - yesterday_hits) / yesterday_hits) * 100, 1)

    last7  = sum(d["total"] for d in daily_history[-7:])
    prev7  = sum(d["total"] for d in daily_history[-14:-7])
    week_delta = round(((last7 - prev7) / prev7) * 100, 1) if prev7 > 0 else None

    last30 = sum(d["total"] for d in daily_history)

    return {
        "today_hits":     today_hits,
        "yesterday_hits": yesterday_hits,
        "today_delta":    today_delta,
        "last7_hits":     last7,
        "prev7_hits":     prev7,
        "week_delta":     week_delta,
        "last30_hits":    last30,
    }


# ─────────────────────────────────────────────────────────────────────────────
# BOUNCE RATE  (previously a nested SQL subquery)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_bounce_rates(session_df: pd.DataFrame) -> Dict[str, float]:
    """
    Input: [{page_url, session_id, hit_count}]
    Returns: {page_url → bounce_rate_percent}
    Bounce = sessions where hit_count == 1.
    """
    if session_df.empty or "page_url" not in session_df.columns:
        return {}

    session_df = session_df.copy()
    session_df["hit_count"] = pd.to_numeric(session_df.get("hit_count", 0), errors="coerce").fillna(0)
    session_df["is_bounce"] = session_df["hit_count"].astype(int) == 1

    grouped = session_df.groupby("page_url").agg(
        total_sessions=("session_id", "count"),
        bounce_count=("is_bounce", "sum"),
    ).reset_index()

    grouped["bounce_rate"] = np.where(
        grouped["total_sessions"] > 0,
        (grouped["bounce_count"] / grouped["total_sessions"] * 100).round(1),
        0.0,
    )

    return dict(zip(grouped["page_url"], grouped["bounce_rate"]))


# ─────────────────────────────────────────────────────────────────────────────
# ERRORS BY PAGE
# ─────────────────────────────────────────────────────────────────────────────

def _aggregate_errors_by_page(err_df: pd.DataFrame) -> Dict[str, Dict]:
    """Returns {url → {error_count, avg_load_time}}"""
    def safe_int(val: Any) -> int:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    if err_df.empty or "url" not in err_df.columns:
        return {}

    agg = err_df.groupby("url").agg(
        error_count=("url", "count"),
        avg_load_time=("load_time_ms", "mean"),
    ).reset_index()

    result = {}
    for _, row in agg.iterrows():
        result[row["url"]] = {
            "error_count":   safe_int(row["error_count"]),
            "avg_load_time": float(row["avg_load_time"]) if pd.notnull(row["avg_load_time"]) else 0.0,
        }
    return result


# ─────────────────────────────────────────────────────────────────────────────
# TOP PAGES  (engagement + bottleneck + sparkline + keyword match)
# ─────────────────────────────────────────────────────────────────────────────

def _score_engagement(avg_duration: float, avg_scroll: float, avg_clicks: float, bounce_rate: float) -> int:
    """
    Engagement score 0-100:
      Dwell       (0-30): 60s+ = max
      Scroll      (0-30): 100% = max
      Interaction (0-25): 5+ clicks = max
      Bounce      (0-15): 0% bounce = max
    """
    dwell       = min((avg_duration / 60) * 30, 30)
    scroll      = (avg_scroll  / 100) * 30
    interaction = min((avg_clicks  / 5)   * 25, 25)
    bounce      = (1 - (bounce_rate / 100)) * 15
    return round(dwell + scroll + interaction + bounce)


def _score_bottleneck(bounce_rate: float, avg_duration: float, error_count: int, avg_load_time: float) -> int:
    """
    Bottleneck score 0-100 (higher = more problematic):
      Bounce   (0-40)
      Dwell    (0-30): low dwell is bad
      Errors   (0-20): each error adds 5 pts
      Load     (0-10)
    """
    b_bounce = (bounce_rate / 100) * 40
    b_dwell  = max(0, (1 - avg_duration / 60) * 30)
    b_error  = min(error_count * 5, 20)
    b_load   = 10 if avg_load_time > 3000 else (5 if avg_load_time > 1500 else 0)
    return round(b_bounce + b_dwell + b_error + b_load)


def _match_keywords(page_url: str, keywords: List[Dict]) -> List[Dict]:
    """Returns keywords whose query text appears in the URL path."""
    try:
        path = urlparse(page_url).path.lower()
    except Exception:
        path = ""
    path_normalized = path.replace("-", " ").replace("_", " ").replace("/", " ")

    matched = []
    for kw in keywords:
        query = str(kw.get("query", "")).lower()
        if query and query in path_normalized:
            matched.append({
                "query":      kw.get("query"),
                "intent":     kw.get("intent"),
                "is_primary": False,
            })
    return matched


def _build_sparkline(page_url: str, spark_df: pd.DataFrame, today_str: str) -> List[int]:
    """Returns a 14-element array of daily hit counts, zero-filled."""
    today = datetime.strptime(today_str, "%Y-%m-%d")

    if spark_df.empty or "page_url" not in spark_df.columns:
        return [0] * 14

    page_spark = spark_df[spark_df["page_url"] == page_url]
    if page_spark.empty:
        return [0] * 14

    page_spark["cnt"] = pd.to_numeric(page_spark.get("cnt", 0), errors="coerce").fillna(0)
    lookup = dict(zip(page_spark["date"].astype(str), page_spark["cnt"].astype(int)))
    series = []
    for i in range(13, -1, -1):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        series.append(lookup.get(d, 0))
    return series


def _build_top_pages(
    page_df: pd.DataFrame,
    spark_df: pd.DataFrame,
    bounce_by_page: Dict[str, float],
    errors_by_page: Dict[str, Dict],
    keywords: List[Dict],
    variant_registry: Dict[str, Dict] = None,
    today_str: str = None,
) -> List[Dict]:
    """Builds the enriched top_pages list the dashboard expects."""
    def safe_int(val: Any) -> int:
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return 0

    if page_df.empty:
        return []

    variant_registry = variant_registry or {}
    
    if today_str:
        today = datetime.strptime(today_str, "%Y-%m-%d")
    else:
        today = datetime.utcnow()

    # ── 1. Vectorized Sparkline Generation ────────────────────────────────────
    # Pivot the spark_df to get [page_url] x [date] matrix
    spark_matrix = pd.DataFrame()
    if not spark_df.empty and "page_url" in spark_df.columns:
        spark_df["cnt"] = pd.to_numeric(spark_df.get("cnt", 0), errors="coerce").fillna(0)
        # Ensure date column is plain string to match strftime lookup keys
        spark_df["date"] = spark_df["date"].astype(str).str[:10]
        spark_matrix = spark_df.pivot_table(
            index="page_url", 
            columns="date", 
            values="cnt", 
            aggfunc="sum"
        ).fillna(0).astype(int)

    # ── 2. Build Result List ──────────────────────────────────────────────────
    def safe_get_str(r, k, default=""):
        val = r.get(k, default)
        if isinstance(val, pd.Series):
            return str(val.iloc[0])
        return str(val)

    result = []
    for _, row in page_df.iterrows():
        page_url    = safe_get_str(row, "page_url", "")
        total_hits  = safe_int(row.get("total_hits"))
        ad_hits     = safe_int(row.get("ad_hits"))
        
        # Extract sparkline from pre-pivoted matrix
        sparkline = []
        if not spark_matrix.empty and page_url in spark_matrix.index:
            row_spark = spark_matrix.loc[page_url]
            for i in range(13, -1, -1):
                d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
                sparkline.append(int(row_spark.get(d, 0)))
        else:
            sparkline = [0] * 14

        avg_duration= float(row.get("avg_duration", 0) or 0)
        avg_scroll  = float(row.get("avg_scroll",   0) or 0)
        avg_clicks  = float(row.get("avg_clicks",   0) or 0)
        today_c     = safe_int(row.get("today_count"))
        yesterday_c = safe_int(row.get("yesterday_count"))

        bounce_rate  = bounce_by_page.get(page_url, 0.0)
        error_info   = errors_by_page.get(page_url, {})
        error_count  = error_info.get("error_count",   0)
        avg_load_time= error_info.get("avg_load_time", 0)

        engagement  = _score_engagement(avg_duration, avg_scroll, avg_clicks, bounce_rate)
        bottleneck  = _score_bottleneck(bounce_rate, avg_duration, error_count, avg_load_time)
        severity    = "critical" if bottleneck >= 60 else ("warning" if bottleneck >= 35 else "good")

        # Delta
        delta_pct = None
        if yesterday_c > 0:
            delta_pct = round(((today_c - yesterday_c) / yesterday_c) * 100, 1)
        elif today_c > 0:
            delta_pct = 100.0

        # Recommendations
        recs = []
        if bounce_rate > 60:    recs.append("High bounce rate — improve page relevance or above-fold content")
        if avg_duration < 20:   recs.append("Low dwell time — add engaging content or video")
        if avg_clicks < 1:      recs.append("Low interaction — add clear CTAs or internal links")
        if error_count > 0:     recs.append(f"{error_count} JS error(s) detected — check browser console")
        if avg_load_time > 3000:
            recs.append(f"Slow page load (avg {round(avg_load_time/1000, 1)}s) — optimise images & scripts")
        elif avg_load_time > 1500:
            recs.append("Moderate load time — consider caching or CDN")

        matched_keywords = _match_keywords(page_url, keywords)
        intents = [k["intent"] for k in matched_keywords if k.get("intent")]
        top_intent = max(set(intents), key=intents.count) if intents else None


        # Variant info from Path Intelligence
        variants = variant_registry.get(page_url, {})
        url_variants = variants.get("url_variants", 1)
        raw_urls     = variants.get("raw_urls", [page_url])

        result.append({
            "page_url":            page_url,
            "display_url":         safe_get_str(row, "display_url", page_url),
            "total_hits":          total_hits,
            "ad_hits":             ad_hits,
            "avg_duration":        round(avg_duration),
            "avg_clicks":          round(avg_clicks, 1),
            "engagement_score":    engagement,
            "is_ad_ready":         engagement >= 70 and total_hits >= 5,
            "today_count":         today_c,
            "yesterday_count":     yesterday_c,
            "delta_pct":           delta_pct,
            "sparkline":           sparkline,
            "matched_keywords":    matched_keywords,
            "top_intent":          top_intent,
            "bounce_rate":         bounce_rate,
            "error_count":         error_count,
            "avg_load_time":       avg_load_time,
            "bottleneck_score":    bottleneck,
            "bottleneck_severity": severity,
            "recommendations":     recs,
            # Path Intelligence fields
            "url_variants":        url_variants,
            "raw_urls":            raw_urls,
            "is_deduplicated":     url_variants > 1,
        })

    return result

# ─────────────────────────────────────────────────────────────────────────────
# TOP REFERRERS
# ─────────────────────────────────────────────────────────────────────────────

def _build_top_referrers(ref_df: pd.DataFrame) -> List[Dict]:
    """Groups referrer URLs by domain and returns top 8."""
    if ref_df.empty or "referrer" not in ref_df.columns:
        return []

    ref_df = ref_df.copy()

    def extract_domain(url: str) -> str:
        try:
            return urlparse(str(url)).hostname or str(url)
        except Exception:
            return str(url)

    ref_df["domain"] = ref_df["referrer"].apply(extract_domain)
    ref_df["domain"] = ref_df["domain"].str.lower()

    grouped = (
        ref_df.groupby("domain")
        .agg(count=("count", "sum"), referrer=("referrer", "first"))
        .reset_index()
        .sort_values("count", ascending=False)
        .head(8)
    )

    return [
        {"domain": row["referrer"], "count": int(row["count"]) if pd.notnull(row["count"]) else 0}
        for _, row in grouped.iterrows()
    ]


# ─────────────────────────────────────────────────────────────────────────────
# TREND VELOCITY
# ─────────────────────────────────────────────────────────────────────────────

def _build_velocity(vel_df: pd.DataFrame) -> Dict[str, List]:
    """Returns rising and falling pages by week-over-week delta."""
    if vel_df.empty or "page_url" not in vel_df.columns:
        return {"rising": [], "falling": []}

    vel_df = vel_df.copy()
    vel_df["last7"] = pd.to_numeric(vel_df.get("last7", 0), errors="coerce").fillna(0).astype(int)
    vel_df["prev7"] = pd.to_numeric(vel_df.get("prev7", 0), errors="coerce").fillna(0).astype(int)

    def delta(row):
        if row["prev7"] > 0:
            return round(((row["last7"] - row["prev7"]) / row["prev7"]) * 100, 1)
        return 100.0 if row["last7"] > 0 else 0.0

    vel_df["delta_pct"] = vel_df.apply(delta, axis=1)
    vel_df = vel_df[vel_df["last7"] > 0]

    rising  = (
        vel_df[vel_df["delta_pct"] > 0]
        .sort_values("delta_pct", ascending=False)
        .head(3)[["page_url", "last7", "prev7", "delta_pct"]]
        .to_dict("records")
    )
    falling = (
        vel_df[vel_df["delta_pct"] < 0]
        .sort_values("delta_pct")
        .head(3)[["page_url", "last7", "prev7", "delta_pct"]]
        .to_dict("records")
    )

    return {"rising": rising, "falling": falling}


# ─────────────────────────────────────────────────────────────────────────────
# GEO BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def _build_geo(geo_df: pd.DataFrame):
    """Returns by_country (top 10), by_city (top 10), by_device."""
    if geo_df.empty:
        return [], [], []

    geo_df = geo_df.copy()
    geo_df["count"] = pd.to_numeric(geo_df.get("count", 0), errors="coerce").fillna(0).astype(int)

    # By country
    by_country = (
        geo_df.groupby("country_code")["count"].sum()
        .reset_index()
        .rename(columns={"country_code": "code"})
        .sort_values("count", ascending=False)
        .head(10)
    )
    by_country["code"] = by_country["code"].fillna("Unknown")
    by_country_list = by_country.to_dict("records")

    # By device
    by_device = (
        geo_df.groupby("device_type")["count"].sum()
        .reset_index()
        .rename(columns={"device_type": "name"})
        .sort_values("count", ascending=False)
    )
    by_device["name"] = by_device["name"].fillna("Desktop")
    by_device_list = by_device.to_dict("records")

    # By city (skip empty city values)
    city_df = geo_df[geo_df["city"].notna() & (geo_df["city"] != "")]
    if not city_df.empty:
        by_city = (
            city_df.groupby("city")["count"].sum()
            .reset_index()
            .rename(columns={"city": "name"})
            .sort_values("count", ascending=False)
            .head(10)
        )
        by_city_list = by_city.to_dict("records")
    else:
        by_city_list = []

    return by_country_list, by_city_list, by_device_list


# ─────────────────────────────────────────────────────────────────────────────
# SITE HEALTH
# ─────────────────────────────────────────────────────────────────────────────

def _build_site_health(err_df: pd.DataFrame) -> Dict:
    """Builds slow_pages, error_type_breakdown, alerts_last_24h."""
    if err_df.empty:
        return {"slow_pages": [], "error_type_breakdown": [], "alerts_last_24h": []}

    err_df = err_df.copy()
    if "load_time_ms" in err_df.columns:
        err_df["load_time_ms"] = pd.to_numeric(err_df["load_time_ms"], errors="coerce").fillna(0)
    else:
        err_df["load_time_ms"] = 0

    # Slow pages
    slow_df = err_df[err_df.get("error_type", pd.Series()) == "slow_load"] if "error_type" in err_df.columns else pd.DataFrame()
    slow_pages = []
    if not slow_df.empty:
        slow_agg = (
            slow_df.groupby("url")
            .agg(avg_load_ms=("load_time_ms", "mean"), count=("url", "count"), last_seen=("created_at", "max"))
            .reset_index()
            .sort_values("avg_load_ms", ascending=False)
            .head(5)
        )
        slow_agg["avg_load_ms"] = pd.to_numeric(slow_agg["avg_load_ms"], errors="coerce").fillna(0).round().astype(int)
        slow_pages = slow_agg.to_dict("records")

    # Error type breakdown
    error_breakdown = []
    if "error_type" in err_df.columns:
        breakdown = (
            err_df.groupby("error_type")["url"].count()
            .reset_index()
            .rename(columns={"error_type": "type", "url": "count"})
            .sort_values("count", ascending=False)
        )
        breakdown["type"] = breakdown["type"].fillna("js_error")
        error_breakdown = breakdown.to_dict("records")

    # Alerts last 24h
    alerts = []
    if "created_at" in err_df.columns:
        try:
            err_df["created_at"] = pd.to_datetime(err_df["created_at"], utc=True)
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=24)
            recent = err_df[err_df["created_at"] >= cutoff]
            if not recent.empty and "error_type" in recent.columns:
                alert_agg = (
                    recent.groupby(["url", "error_type"])["url"].count()
                    .reset_index(name="count")
                    .sort_values("count", ascending=False)
                    .head(5)
                )
                alerts = alert_agg.to_dict("records")
        except Exception as e:
            logger.warning(f"Could not parse created_at for alerts: {e}")

    return {
        "slow_pages":           slow_pages,
        "error_type_breakdown": error_breakdown,
        "alerts_last_24h":      alerts,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DEEP PATH-LEVEL SITE ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_path(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes deep path-level analytics for a specific URL using Pandas.
    Surgically offloaded from DeepAnalyticsController.php to maximize speed.
    """
    path = data.get("path", "")
    period = int(data.get("period", 30))
    exclude_bots = bool(data.get("exclude_bots", True))
    raw_events = data.get("events", [])
    errors = data.get("errors", [])
    returning_rate = float(data.get("returning_rate", 0.0))
    
    meta = data.get("meta", {})
    today_str = meta.get("today", datetime.utcnow().strftime("%Y-%m-%d"))
    today = datetime.strptime(today_str, "%Y-%m-%d")

    df = pd.DataFrame(raw_events) if raw_events else pd.DataFrame()

    if df.empty or "page_url" not in df.columns:
        # Default empty response structure matching PHP exactly
        return {
            "path": path,
            "period": period,
            "summary": {
                "total_visits": 0,
                "unique_sessions": 0,
                "avg_dwell": 0,
                "avg_scroll": 0.0,
                "avg_clicks": 0.0,
                "ad_hits": 0,
                "hot_leads": 0,
                "warm_leads": 0,
                "cold_leads": 0,
                "bounce_rate": 0.0,
                "entry_rate": 0.0,
                "exit_rate": 0.0,
                "engagement_score": 0,
                "bottleneck_score": 0,
                "bottleneck_severity": "good",
                "avg_load_ms": 0,
                "ad_ready": False,
                "top_intent": None,
                "returning_rate": returning_rate,
            },
            "by_country": [],
            "by_device": [],
            "by_browser": [],
            "by_city": [],
            "referrers": [],
            "utm_breakdown": [],
            "prev_pages": [],
            "next_pages": [],
            "hour_pattern": [{"hour": h, "visits": 0, "avg_dwell": 0.0} for h in range(24)],
            "dow_pattern": [{"day": d, "visits": 0} for d in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]],
            "daily_history": [],
            "errors": errors,
            "recommendations": []
        }

    # ── 1. Subset path events ───────────────────────────────────────────────
    df_path = df[df["page_url"] == path]
    
    total_visits = len(df_path)
    unique_sessions = df_path["session_id"].nunique() if "session_id" in df_path.columns else 0
    avg_dwell = float(df_path["duration_seconds"].mean()) if "duration_seconds" in df_path.columns else 0.0
    avg_scroll = float(df_path["max_scroll_depth"].mean()) if "max_scroll_depth" in df_path.columns else 0.0
    avg_clicks = float(df_path["click_count"].mean()) if "click_count" in df_path.columns else 0.0
    
    # Fill NAs
    avg_dwell = 0.0 if math.isnan(avg_dwell) else avg_dwell
    avg_scroll = 0.0 if math.isnan(avg_scroll) else avg_scroll
    avg_clicks = 0.0 if math.isnan(avg_clicks) else avg_clicks

    # Ad hits
    ad_cond = pd.Series([False] * len(df_path), index=df_path.index)
    for col in ["gclid", "utm_campaign", "google_campaign_id"]:
        if col in df_path.columns:
            ad_cond = ad_cond | (df_path[col].notnull() & (df_path[col] != ""))
    ad_hits = int(ad_cond.sum())

    # Leads (hot, warm, cold)
    hot_cond = pd.Series([False] * len(df_path), index=df_path.index)
    warm_cond = pd.Series([False] * len(df_path), index=df_path.index)
    cold_cond = pd.Series([False] * len(df_path), index=df_path.index)

    if "duration_seconds" in df_path.columns:
        duration = df_path["duration_seconds"].fillna(0)
        clicks = df_path["click_count"].fillna(0) if "click_count" in df_path.columns else pd.Series([0]*len(df_path), index=df_path.index)
        
        hot_cond = (duration >= 90) | ((duration >= 60) & (clicks >= 2))
        warm_cond = ((duration >= 30) | (clicks >= 1)) & (~hot_cond)
        cold_cond = (duration < 30) & (clicks < 1)

    hot_leads = int(hot_cond.sum())
    warm_leads = int(warm_cond.sum())
    cold_leads = int(cold_cond.sum())

    # ── 2. Bounce Rate ──────────────────────────────────────────────────────
    bounce_rate = 0.0
    entry_rate = 0.0
    exit_rate = 0.0
    prev_pages = []
    next_pages = []

    if "session_id" in df_path.columns and not df_path["session_id"].dropna().empty:
        visited_sessions = df_path["session_id"].dropna().unique()
        total_sessions = len(visited_sessions)
        
        if total_sessions > 0:
            df_sess = df[df["session_id"].isin(visited_sessions)].copy()
            df_sess["created_at_dt"] = pd.to_datetime(df_sess["created_at"])
            df_sess = df_sess.sort_values("created_at_dt")

            # Unique pages per session
            pages_per_sess = df_sess.groupby("session_id")["page_url"].nunique()
            bounces = int((pages_per_sess == 1).sum())
            bounce_rate = round((bounces / total_sessions) * 100, 1)

            # Entry / Exit Rates
            first_hits = df_sess.groupby("session_id").first()
            last_hits = df_sess.groupby("session_id").last()
            
            entry_count = int((first_hits["page_url"] == path).sum())
            exit_count = int((last_hits["page_url"] == path).sum())
            
            entry_rate = round((entry_count / total_sessions) * 100, 1)
            exit_rate = round((exit_count / total_sessions) * 100, 1)

            # Previous & Next Flows (Entry & Exit Flows)
            from collections import Counter
            prev_list = []
            next_list = []

            for _, cur in df_path.iterrows():
                sess_id = cur.get("session_id")
                cur_time = pd.to_datetime(cur.get("created_at"))
                if not sess_id or pd.isna(cur_time):
                    continue
                
                sess_df = df_sess[df_sess["session_id"] == sess_id]
                
                prev_evts = sess_df[(sess_df["created_at_dt"] < cur_time) & (sess_df["page_url"] != path)]
                next_evts = sess_df[(sess_df["created_at_dt"] > cur_time) & (sess_df["page_url"] != path)]
                
                prev_list.extend(prev_evts["page_url"].dropna().tolist())
                next_list.extend(next_evts["page_url"].dropna().tolist())

            prev_counts = Counter(prev_list)
            next_counts = Counter(next_list)

            prev_pages = [{"page_url": k, "count": v} for k, v in prev_counts.most_common(5)]
            next_pages = [{"page_url": k, "count": v} for k, v in next_counts.most_common(5)]

    # ── 3. Geo, Device, Browser breakdowns ──────────────────────────────────
    by_country = []
    if "country_code" in df_path.columns:
        by_country = (
            df_path.groupby("country_code")["page_url"].count()
            .reset_index(name="count")
            .rename(columns={"country_code": "code"})
            .sort_values("count", ascending=False)
            .head(10).to_dict("records")
        )
        for c in by_country:
            if not c["code"]:
                c["code"] = "Unknown"

    by_city = []
    if "city" in df_path.columns:
        by_city = (
            df_path[df_path["city"].notnull() & (df_path["city"] != "")]
            .groupby("city")["page_url"].count()
            .reset_index(name="count")
            .rename(columns={"city": "name"})
            .sort_values("count", ascending=False)
            .head(8).to_dict("records")
        )

    by_device = []
    if "device_type" in df_path.columns:
        by_device = (
            df_path.groupby("device_type")["page_url"].count()
            .reset_index(name="count")
            .rename(columns={"device_type": "name"})
            .sort_values("count", ascending=False)
            .to_dict("records")
        )

    by_browser = []
    if "browser" in df_path.columns:
        by_browser = (
            df_path.groupby("browser")["page_url"].count()
            .reset_index(name="count")
            .rename(columns={"browser": "name"})
            .sort_values("count", ascending=False)
            .head(5).to_dict("records")
        )

    # ── 4. Referrers ────────────────────────────────────────────────────────
    referrers_list = []
    if "referrer" in df_path.columns:
        df_ref = df_path[df_path["referrer"].notnull() & (df_path["referrer"] != "")].copy()
        if not df_ref.empty:
            def get_domain(url):
                try:
                    host = urlparse(url).netloc
                    return host if host else url
                except Exception:
                    return url
            df_ref["domain"] = df_ref["referrer"].apply(get_domain).str.lower()
            ref_grouped = df_ref.groupby("domain")["page_url"].count().reset_index(name="count").sort_values("count", ascending=False).head(6)
            
            for _, r in ref_grouped.iterrows():
                orig = df_ref[df_ref["domain"] == r["domain"]]["referrer"].iloc[0]
                referrers_list.append({"domain": orig, "count": int(r["count"])})

    # ── 5. UTM / Attribution ────────────────────────────────────────────────
    utm_breakdown = []
    utm_cols = ["utm_source", "utm_medium", "utm_campaign"]
    if all(col in df_path.columns for col in utm_cols):
        utm_df = df_path[df_path["utm_source"].notnull() | df_path["utm_medium"].notnull() | df_path["utm_campaign"].notnull()].copy()
        if not utm_df.empty:
            utm_grouped = utm_df.groupby(utm_cols, dropna=False)["page_url"].count().reset_index(name="visits").sort_values("visits", ascending=False).head(8)
            utm_grouped = utm_grouped.fillna("—")
            utm_breakdown = utm_grouped.rename(columns={"utm_source": "source", "utm_medium": "medium", "utm_campaign": "campaign"}).to_dict("records")

    # ── 6. Hour and DOW patterns ────────────────────────────────────────────
    hour_pattern = []
    dow_pattern = []
    if "created_at" in df_path.columns:
        df_path["created_at_dt"] = pd.to_datetime(df_path["created_at"])
        
        # Hour of day
        df_path["hour"] = df_path["created_at_dt"].dt.hour
        hour_grouped = df_path.groupby("hour").agg(
            visits=("page_url", "count"),
            avg_dwell=("duration_seconds", "mean")
        ).fillna(0).to_dict("index")
        
        for h in range(24):
            g = hour_grouped.get(h, {"visits": 0, "avg_dwell": 0.0})
            hour_pattern.append({
                "hour": h,
                "visits": int(g["visits"]),
                "avg_dwell": float(g["avg_dwell"]),
            })

        # Day of week (1=Sun ... 7=Sat)
        df_path["dow"] = df_path["created_at_dt"].dt.dayofweek # Mon=0 ... Sun=6
        df_path["dow"] = (df_path["dow"] + 1) % 7 + 1
        dow_grouped = df_path.groupby("dow")["page_url"].count().fillna(0).to_dict()
        
        days_names = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        for d in range(1, 8):
            dow_pattern.append({
                "day": days_names[d - 1],
                "visits": int(dow_grouped.get(d, 0)),
            })

    # ── 7. Daily History ────────────────────────────────────────────────────
    daily_history = []
    if "created_at" in df_path.columns:
        df_path["date"] = df_path["created_at_dt"].dt.strftime("%Y-%m-%d")
        
        # Build ad condition matrix
        ad_hits_by_row = pd.Series([0] * len(df_path), index=df_path.index)
        for col in ["gclid", "utm_campaign", "google_campaign_id"]:
            if col in df_path.columns:
                ad_hits_by_row = ad_hits_by_row | (df_path[col].notnull() & (df_path[col] != ""))
        
        df_path["is_ad_hit"] = ad_hits_by_row.astype(int)
        
        daily_grouped = df_path.groupby("date").agg(
            visits=("page_url", "count"),
            ad_hits=("is_ad_hit", "sum")
        ).fillna(0).to_dict("index")

        for i in range(period - 1, -1, -1):
            date_dt = today - timedelta(days=i)
            d_str = date_dt.strftime("%Y-%m-%d")
            lbl = date_dt.strftime("%b %e").replace("  ", " ")
            g = daily_grouped.get(d_str, {"visits": 0, "ad_hits": 0})
            daily_history.append({
                "date": d_str,
                "label": lbl,
                "visits": int(g["visits"]),
                "ad_hits": int(g["ad_hits"]),
            })

    # ── 8. JS Errors calculation ──────────────────────────────────────────
    error_count = len(errors)
    avg_load_ms = 0.0
    if errors:
        avg_load_ms = float(pd.DataFrame(errors)["load_time_ms"].mean())
        avg_load_ms = 0.0 if math.isnan(avg_load_ms) else avg_load_ms

    # ── 9. Engagement / Bottleneck Scoring ─────────────────────────────────
    dwell_score     = min((avg_dwell / 60.0) * 30.0, 30.0)
    scroll_score    = (avg_scroll / 100.0) * 30.0
    interact_score  = min((avg_clicks / 5.0) * 25.0, 25.0)
    bounce_score    = (1.0 - (bounce_rate / 100.0)) * 15.0
    engagement_score = round(dwell_score + scroll_score + interact_score + bounce_score)

    bounce_b = bounce_rate / 100.0 * 40.0
    dwell_b  = max(0.0, (1.0 - avg_dwell / 60.0) * 30.0)
    error_b  = min(error_count * 5.0, 20.0)
    load_b   = 10.0 if avg_load_ms > 3000 else (5.0 if avg_load_ms > 1500 else 0.0)
    bottleneck_score    = round(bounce_b + dwell_b + error_b + load_b)
    bottleneck_severity = "critical" if bottleneck_score >= 60 else ("warning" if bottleneck_score >= 35 else "good")

    # ── 10. Keyword intent matching ─────────────────────────────────────────
    matched_keywords = []
    try:
        path_lower = urlparse(path).path.lower() if urlparse(path).path else path.lower()
    except Exception:
        path_lower = path.lower()
    path_normalized = path_lower.replace("-", " ").replace("_", " ").replace("/", " ")
    
    for kw in data.get("keywords", []):
        q = str(kw.get("query", "")).lower()
        if q and q in path_normalized:
            matched_keywords.append({
                "query": kw.get("query"),
                "intent": kw.get("intent"),
                "is_primary": False,
            })
    
    intents = [k["intent"] for k in matched_keywords if k.get("intent")]
    top_intent = max(set(intents), key=intents.count) if intents else None

    # ── 11. Marketing Recommendations ───────────────────────────────────────
    recs = []
    if bounce_rate > 60:
        recs.append({"type": "warning", "text": "High bounce rate — improve above-fold content and page relevance"})
    if avg_dwell < 20:
        recs.append({"type": "warning", "text": "Low dwell time — add engaging media or interactive CTAs"})
    if avg_clicks < 1:
        recs.append({"type": "warning", "text": "Low interaction — add clear call-to-action buttons"})
    if error_count > 0:
        recs.append({"type": "critical", "text": f"{error_count} JS error(s) detected — fix to prevent lead drop-off"})
    if avg_load_ms > 3000:
        recs.append({"type": "critical", "text": f"Slow page load ({round(avg_load_ms/1000, 1)}s) — compress images & defer scripts"})
    if entry_rate > 40:
        recs.append({"type": "success", "text": "Strong entry page — ideal for paid ad landing. Consider A/B testing CTAs"})
    if exit_rate > 50:
        recs.append({"type": "warning", "text": "High exit rate — add exit-intent offers or internal link suggestions"})
    
    ad_ready = engagement_score >= 65 and total_visits >= 5
    if ad_ready:
        recs.append({"type": "success", "text": "Ad-ready page — engagement score qualifies for targeted paid campaigns"})
    
    if top_intent == "commercial":
        recs.append({"type": "success", "text": "Commercial intent detected — Strategy: Use high-bid Search Ads with 'Buy Now' or 'Get Quote' copy"})
        recs.append({"type": "success", "text": "Audience Strategy: Target 'In-Market' segments related to " + (matched_keywords[0]["query"] if matched_keywords else "your niche")})
    elif top_intent == "informational":
        recs.append({"type": "success", "text": "Informational intent detected — Strategy: Best for Awareness/Display ads or Content Marketing retargeting"})
        recs.append({"type": "success", "text": "Engagement Strategy: Use 'Learn More' or 'Download Guide' to capture top-funnel interest"})
    
    if avg_dwell > 60 and bounce_rate < 30:
        recs.append({"type": "success", "text": "Viral potential — High dwell and low bounce suggest this content is highly sticky. Amplify via Social Ads."})

    return {
        "path": path,
        "period": period,
        "summary": {
            "total_visits": total_visits,
            "unique_sessions": unique_sessions,
            "avg_dwell": round(avg_dwell),
            "avg_scroll": round(avg_scroll, 1),
            "avg_clicks": round(avg_clicks, 1),
            "ad_hits": ad_hits,
            "hot_leads": hot_leads,
            "warm_leads": warm_leads,
            "cold_leads": cold_leads,
            "bounce_rate": bounce_rate,
            "entry_rate": entry_rate,
            "exit_rate": exit_rate,
            "engagement_score": engagement_score,
            "bottleneck_score": bottleneck_score,
            "bottleneck_severity": bottleneck_severity,
            "avg_load_ms": round(avg_load_ms),
            "ad_ready": ad_ready,
            "top_intent": top_intent,
            "returning_rate": returning_rate,
        },
        "by_country": by_country,
        "by_device": by_device,
        "by_browser": by_browser,
        "by_city": by_city,
        "referrers": referrers_list,
        "utm_breakdown": utm_breakdown,
        "prev_pages": prev_pages,
        "next_pages": next_pages,
        "hour_pattern": hour_pattern,
        "dow_pattern": dow_pattern,
        "daily_history": daily_history,
        "errors": errors,
        "recommendations": recs
    }
