from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
import math

app = FastAPI(title="Metapilot Analytical Engine")

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

def optimize_budget(campaign_data: List[GoogleAdsEntry]) -> List[Dict]:
    """
    Calculates ROAS and suggests budget reallocation.
    """
    df = pd.DataFrame([c.dict() for c in campaign_data])
    if df.empty:
        return []
        
    # Aggregate by campaign
    agg = df.groupby('campaign_name').agg({
        'cost': 'sum',
        'conversions': 'sum',
        'clicks': 'sum',
        'impressions': 'sum'
    }).reset_index()
    
    agg['roas'] = agg['conversions'] / agg['cost'].replace(0, np.inf)
    agg['cpc'] = agg['cost'] / agg['clicks'].replace(0, np.inf)
    
    total_spend = agg['cost'].sum()
    if total_spend == 0:
        return []

    mean_roas = agg['roas'].replace([np.inf, -np.inf], 0).mean()
    
    recommendations = []
    for _, row in agg.iterrows():
        roas = row['roas']
        if roas > mean_roas * 1.2:
            action = "Increase Budget"
            reason = f"ROAS ({roas:.2f}) is 20% above average."
        elif roas < mean_roas * 0.8 and roas != 0:
            action = "Decrease Budget"
            reason = f"ROAS ({roas:.2f}) is tracking below average."
        else:
            action = "Maintain"
            reason = "Performing within expected range."
            
        recommendations.append({
            "campaign": row['campaign_name'],
            "current_spend": round(row['cost'], 2),
            "roas": round(roas, 2) if roas != np.inf else 0,
            "action": action,
            "reason": reason
        })
        
    return recommendations

# --- Endpoints ---

@app.post("/predict/ad-performance")
async def predict_ad_performance(prompt: AdPerformancePrompt):
    """
    Calculates ROI forecasts and budget optimization.
    """
    if not prompt.campaign_data:
        raise HTTPException(status_code=400, detail="No campaign data provided")
        
    recommendations = optimize_budget(prompt.campaign_data)
    
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

@app.post("/predict/full")

def calculate_propensity_score(channels_data: Dict[str, ChannelData], returning_users: int, sessions: int) -> Dict[str, float]:
    """
    Implements: P(Conversion) = 1 / (1 + e^-(beta0 + beta1*ChannelScore + beta2*ln(Sessions) + beta3*EngagementRate))
    Simplified for this version to provide a relative probability score per channel.
    """
    # Weights (beta coefficients - simplified)
    beta0 = -2.0  # Base intercept
    beta1 = 0.5   # Channel weight
    beta2 = 0.3   # Session weight (log)
    
    # Predefined channel quality scores (could be learned in future versions)
    channel_quality = {
        "Paid Search": 0.8,
        "Organic Search": 0.7,
        "Direct": 0.5,
        "Referral": 0.6,
        "Social": 0.4,
        "Email": 0.75
    }

    scores = {}
    for channel, data in channels_data.items():
        if data.users == 0:
            continue
            
        qual = channel_quality.get(channel, 0.5)
        # Logarithmic session weight to handle high-freq users
        session_factor = math.log(sessions + 1)
        
        # Logistic function
        z = beta0 + (beta1 * qual) + (beta2 * session_factor)
        probability = 1 / (1 + math.exp(-z))
        
        scores[channel] = round(probability, 4)
        
    return scores

def detect_source_fatigue(history: List[HistoricalEntry], forecast_days: int) -> Dict[str, Dict]:
    """
    Uses Prophet to forecast session trends and detect diminishing returns.
    """
    # Prepare data for all sources
    source_stats = {}
    all_dates = [entry.date for entry in history]
    
    # Extract unique sources
    unique_sources = set()
    for entry in history:
        unique_sources.update(entry.sources.keys())
        
    for source in unique_sources:
        df_source = pd.DataFrame([
            {"ds": entry.date, "y": entry.sources.get(source, 0)}
            for entry in history
        ])
        
        if len(df_source) < 7: # Need at least a week of data
            continue
            
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df_source)
        
        future = m.make_future_dataframe(periods=int(forecast_days))
        forecast = m.predict(future)
        
        # Detect fatigue: Is the trend negative?
        # Calculate slope of the forecast
        last_val = forecast['yhat'].iloc[-1]
        prev_val = forecast['yhat'].iloc[-int(forecast_days)]
        
        trend = (last_val - prev_val) / prev_val if prev_val > 0 else 0
        
        source_stats[source] = {
            "forecasted_sessions": round(last_val, 2),
            "trend_percentage": round(trend * 100, 2),
            "is_fatigued": trend < -0.15, # 15% drop predicted
            "confidence": 0.85 # Placeholder
        }
        
    return source_stats

# --- Endpoints ---

@app.post("/predict/full")
async def predict_full(prompt: AnalyticsPrompt):
    """
    Comprehensive endpoint that returns Propensity, Fatigue, and Rankings.
    """
    if not prompt.historical_data:
        raise HTTPException(status_code=400, detail="Insufficient historical data")

    # 1. Lead Propensity
    latest_entry = prompt.historical_data[-1]
    propensity = calculate_propensity_score(
        latest_entry.channels, 
        latest_entry.returning_users, 
        latest_entry.sessions
    )

    # 2. Source Fatigue
    fatigue = detect_source_fatigue(
        prompt.historical_data, 
        prompt.config.get("forecast_days", 14)
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

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
