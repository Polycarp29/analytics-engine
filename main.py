from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import math
import logging
import traceback

app = FastAPI(title="Metapilot Analytical Engine")

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

class AdCampaignData(BaseModel):
    name: str
    total_cost: float
    total_clicks: int
    total_impressions: int
    total_conversions: int
    keywords: Optional[List[str]] = []

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
    google_ads_data: Optional[List[AdCampaignData]] = []
    by_country: Optional[List[GeoEntry]] = []
    by_city: Optional[List[GeoEntry]] = []
    top_queries: Optional[List[GscQueryEntry]] = []
    top_pages: Optional[List[GscPageEntry]] = []
    config: Dict[str, float] = Field(default_factory=lambda: {"forecast_days": 90, "propensity_threshold": 0.75})

import engine

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

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
