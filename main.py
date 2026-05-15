from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import math
import logging
import traceback

app = FastAPI(title="Metapilot Analytical Engine")

@app.get("/")
async def root():
    return {
        "service": "Metapilot Analytical Engine",
        "version": "1.2.0",
        "active_modules": ["cdn_intelligence", "keyword_research", "forecast_engine"],
        "endpoints": ["/analyze/cdn", "/health/cdn", "/predict/ad-performance", "/predict/full"]
    }


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Models ---

class ChannelData(BaseModel):
    users: int
    conversions: int

class GscMetrics(BaseModel):
    clicks: int
    impressions: int
    position: float

class GoogleAdsEntry(BaseModel):
    date: str
    campaign_name: str
    clicks: int
    conversions: int
    cost: float
    impressions: int
    keywords: Optional[List[str]] = []

class AdPerformancePrompt(BaseModel):
    property_id: str
    campaign_data: List[GoogleAdsEntry]

class HistoricalEntry(BaseModel):
    date: str
    users: int
    new_users: int
    returning_users: int
    sessions: int
    conversions: int
    bounce_rate: float
    avg_session_duration: float
    channels: Dict[str, ChannelData]
    sources: Dict[str, int]
    gsc_metrics: Optional[GscMetrics] = None


class GeoEntry(BaseModel):
    name: str
    activeUsers: int

class GscQueryEntry(BaseModel):
    name: str
    clicks: int
    impressions: int
    position: float
    ctr: float

class GscPageEntry(BaseModel):
    name: str
    clicks: int
    impressions: int
    position: float

class AnalyticsPrompt(BaseModel):
    property_id: str
    property_name: Optional[str] = "Unknown Property"
    period_start: str
    period_end: str
    historical_data: List[HistoricalEntry]
    google_ads_data: Optional[List[GoogleAdsEntry]] = []
    by_country: Optional[List[GeoEntry]] = []
    by_city: Optional[List[GeoEntry]] = []
    top_queries: Optional[List[GscQueryEntry]] = []
    top_pages: Optional[List[GscPageEntry]] = []
    config: Dict[str, float] = Field(default_factory=lambda: {"forecast_days": 90, "propensity_threshold": 0.75})

import engine
import global_trends_crawler

class GlobalTrendsRequest(BaseModel):
    geo: Optional[str] = "KE"
    niches: Optional[List[str]] = []

@app.post("/trends/global")
async def get_global_trends(prompt: GlobalTrendsRequest):
    try:
        # Get Serper API key from request or env (prefer request for Laravel flexibility)
        api_key = os.getenv("SERPER_API", "5c978de70112aba82cc95c4b7c7e5dadd472d54d")
        
        trends = global_trends_crawler.discover_global_trends(
            api_key=api_key,
            geo=prompt.geo,
            niches=prompt.niches
        )
        return {
            "trends": trends,
            "count": len(trends),
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in global-trends for geo {prompt.geo}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/ad-performance")
async def predict_ad_performance(prompt: AdPerformancePrompt):
    # This endpoint is kept for legacy/specific ad tasks
    try:
        if not prompt.campaign_data:
            raise HTTPException(status_code=400, detail="No campaign data provided")
            
        campaign_dicts = [c.model_dump() for c in prompt.campaign_data]
        recommendations = engine.optimize_budget(campaign_dicts)
        
        return {
            "property_id": prompt.property_id,
            "recommendations": recommendations
        }
    except Exception as e:
        logger.error(f"Error in predict_ad_performance: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/full")
async def predict_full(prompt: AnalyticsPrompt):
    """
    Strategic endpoint: Correlates GA4, GSC, and Ads data to generate 
    forecasts and actionable business recommendations.
    """
    try:
        if not prompt.historical_data:
            raise HTTPException(status_code=400, detail="Insufficient historical data")

        # Convert pydantic models to dicts for engine
        data_dict = prompt.model_dump()
        
        # Call the new strategic analysis engine
        analysis_result = engine.generate_strategic_analysis(data_dict)

        return {
            "property_id": prompt.property_id,
            "generated_at": datetime.now().isoformat(),
            "valid_until": (datetime.now() + timedelta(days=2)).isoformat(),
            **analysis_result
        }
    except Exception as e:
        logger.error(f"Error in predict_full for property {prompt.property_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error in strategic logic: {str(e)}")

class KeywordHistoryItem(BaseModel):
    date: str
    interest_value: float

class KeywordDecayRequest(BaseModel):
    keyword_id: int
    history: List[KeywordHistoryItem]

@app.post("/predict/keyword-decay")
async def predict_keyword_decay(prompt: KeywordDecayRequest):
    try:
        # Convert pydantic models to dicts for engine
        history_dicts = [item.model_dump() for item in prompt.history]
        
        # Call the engine logic
        result = engine.predict_keyword_decay(history_dicts)
        
        return {
            "keyword_id": prompt.keyword_id,
            "prediction": result,
            "generated_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error in keyword-decay for ID {prompt.keyword_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ─────────────────────────────────────────────────────────────────────────────
# CDN ANALYTICS ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

import cdn_processor

class CdnDailySummaryItem(BaseModel):
    date: str
    total: int = 0
    ad_hits: int = 0

class CdnPageStatItem(BaseModel):
    page_url: str
    total_hits: int = 0
    avg_duration: Optional[float] = 0
    avg_scroll: Optional[float] = 0
    avg_clicks: Optional[float] = 0
    ad_hits: int = 0
    today_count: int = 0
    yesterday_count: int = 0

class CdnSessionCountItem(BaseModel):
    page_url: str
    session_id: Optional[str] = None
    hit_count: int = 1


class CdnSparklineItem(BaseModel):
    page_url: str
    date: str
    cnt: int = 0

class CdnVelocityItem(BaseModel):
    page_url: str
    last7: int = 0
    prev7: int = 0

class CdnGeoItem(BaseModel):
    country_code: Optional[str] = None
    city: Optional[str] = None
    device_type: Optional[str] = None
    count: int = 0

class CdnReferrerItem(BaseModel):
    referrer: str
    count: int = 0

class CdnErrorItem(BaseModel):
    url: str
    error_type: Optional[str] = None
    load_time_ms: Optional[float] = None
    created_at: Optional[str] = None

class CdnKeywordItem(BaseModel):
    query: str
    intent: Optional[str] = None

class CdnMeta(BaseModel):
    today: Optional[str] = None
    yesterday: Optional[str] = None
    pages_page: int = 1
    pages_per_page: int = 10
    pages_total: int = 0
    normalize_ids: bool = False


class CdnAnalyticsRequest(BaseModel):
    org_id: int
    site_id: Optional[int] = None
    daily_summary:  List[CdnDailySummaryItem]  = []
    page_stats:     List[CdnPageStatItem]       = []
    session_counts: List[CdnSessionCountItem]   = []
    sparkline_raw:  List[CdnSparklineItem]      = []
    velocity_raw:   List[CdnVelocityItem]       = []
    geo_raw:        List[CdnGeoItem]            = []
    referrers:      List[CdnReferrerItem]       = []
    errors:         List[CdnErrorItem]          = []
    keywords:       List[CdnKeywordItem]        = []
    meta:           CdnMeta                     = CdnMeta()

@app.post("/analyze/cdn")
async def analyze_cdn(payload: CdnAnalyticsRequest):
    """
    Receives pre-aggregated CDN data from Laravel and returns
    the full analytics payload (engagement scores, bounce rates,
    bottleneck scores, geo breakdown, trend velocity, site health).
    Response shape is identical to what CdnTrackingController@analytics
    previously returned, so the Vue dashboard requires zero changes.
    """
    try:
        data = payload.model_dump()
        result = cdn_processor.analyze(data)
        return result
    except Exception as e:
        logger.error(f"Error in /analyze/cdn for org {payload.org_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"CDN analytics engine error: {str(e)}")

@app.get("/health/cdn")
async def health_cdn():
    return {
        "status":       "healthy",
        "module":       "cdn_processor",
        "pandas":       pd.__version__,
        "numpy":        np.__version__,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
