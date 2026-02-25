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

class HistoricalEntry(BaseModel):
    date: str
    returning_users: int
    sessions: int
    conversions: int
    channels: Dict[str, ChannelData]
    sources: Dict[str, int]

class GoogleAdsEntry(BaseModel):
    date: str
    campaign_name: str
    clicks: int
    conversions: int
    cost: float
    impressions: int

class KeywordTrend(BaseModel):
    keyword: str
    trend_score: float # e.g. growth rate

class AdPerformancePrompt(BaseModel):
    property_id: str
    campaign_data: List[GoogleAdsEntry]
    keyword_trends: Optional[List[KeywordTrend]] = []

class AnalyticsPrompt(BaseModel):
    property_id: str
    historical_data: List[HistoricalEntry]
    google_ads_data: Optional[List[GoogleAdsEntry]] = []
    config: Dict[str, float] = Field(default_factory=lambda: {"forecast_days": 14, "propensity_threshold": 0.75})

import engine

@app.post("/predict/ad-performance")
async def predict_ad_performance(prompt: AdPerformancePrompt):
    """
    Calculates ROI forecasts and budget optimization.
    """
    try:
        if not prompt.campaign_data:
            raise HTTPException(status_code=400, detail="No campaign data provided")
            
        campaign_dicts = [c.dict() for c in prompt.campaign_data]
        recommendations = engine.optimize_budget(campaign_dicts)
        
        # Interlink with keyword trends if provided
        for rec in recommendations:
            campaign_name = rec['campaign'].lower()
            correlated_trends = [
                t for t in prompt.keyword_trends 
                if t.keyword.lower() in campaign_name or campaign_name in t.keyword.lower()
            ]
            
            if correlated_trends:
                top_trend = max(correlated_trends, key=lambda x: x.trend_score)
                if top_trend.trend_score > 20: # 20% growth
                    rec['action'] = "Scale Up (High Intent)"
                    rec['reason'] += f" High keyword growth ({top_trend.trend_score}%) detected for '{top_trend.keyword}'."

        return {
            "property_id": prompt.property_id,
            "recommendations": recommendations,
            "forecasted_roas_impact": 0.15 # Placeholder for expected lift
        }
    except Exception as e:
        logger.error(f"Error in predict_ad_performance for property {prompt.property_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error in prediction logic: {str(e)}")

@app.post("/predict/full")
async def predict_full(prompt: AnalyticsPrompt):
    """
    Comprehensive endpoint that returns Propensity, Fatigue, and Rankings.
    """
    try:
        if not prompt.historical_data:
            raise HTTPException(status_code=400, detail="Insufficient historical data")

        # 1. Lead Propensity
        latest_entry = prompt.historical_data[-1]
        channels_dict = {k: v.dict() for k, v in latest_entry.channels.items()}
        
        propensity = engine.calculate_propensity_score(
            channels_dict, 
            latest_entry.returning_users, 
            latest_entry.sessions
        )

        # 2. Source Fatigue
        history_dicts = [e.dict() for e in prompt.historical_data]
        fatigue = engine.detect_source_fatigue(
            history_dicts, 
            int(prompt.config.get("forecast_days", 14))
        )

        # 3. Cross-Channel Ranking
        rankings = []
        for channel, prob in propensity.items():
            data = latest_entry.channels.get(channel)
            rankings.append({
                "channel": channel,
                "propensity": prob,
                "efficiency_index": round(prob * (data.conversions / data.users if data.users > 0 else 0), 4)
            })
        
        # Sort by efficiency
        rankings = sorted(rankings, key=lambda x: x['efficiency_index'], reverse=True)

        return {
            "property_id": prompt.property_id,
            "generated_at": datetime.now().isoformat(),
            "valid_until": (datetime.now() + timedelta(days=1)).isoformat(),
            "predictions": {
                "propensity_scores": propensity,
                "source_fatigue": fatigue,
                "performance_rankings": rankings
            }
        }
    except Exception as e:
        logger.error(f"Error in predict_full for property {prompt.property_id}: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Internal Server Error in prediction logic: {str(e)}")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
