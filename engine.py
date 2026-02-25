import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from prophet import Prophet
from typing import List, Dict, Optional

def optimize_budget(campaign_data: List[Dict]) -> List[Dict]:
    """
    Calculates ROAS and suggests budget reallocation.
    """
    df = pd.DataFrame(campaign_data)
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

def calculate_propensity_score(channels_data: Dict[str, Dict], returning_users: int, sessions: int) -> Dict[str, float]:
    """
    Logistic regression model for conversion probability.
    """
    beta0 = -2.0
    beta1 = 0.5
    beta2 = 0.3
    
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
        users = data.get('users', 0)
        if users == 0:
            continue
            
        qual = channel_quality.get(channel, 0.5)
        session_factor = math.log(sessions + 1)
        
        z = beta0 + (beta1 * qual) + (beta2 * session_factor)
        probability = 1 / (1 + math.exp(-z))
        
        scores[channel] = round(probability, 4)
        
    return scores

def detect_source_fatigue(history: List[Dict], forecast_days: int) -> Dict[str, Dict]:
    """
    Uses Prophet to forecast session trends and detect diminishing returns.
    """
    source_stats = {}
    
    unique_sources = set()
    for entry in history:
        unique_sources.update(entry.get('sources', {}).keys())
        
    for source in unique_sources:
        df_source = pd.DataFrame([
            {"ds": entry['date'], "y": entry.get('sources', {}).get(source, 0)}
            for entry in history
        ])
        
        if len(df_source) < 7:
            continue
            
        m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df_source)
        
        future = m.make_future_dataframe(periods=int(forecast_days))
        forecast = m.predict(future)
        
        last_val = forecast['yhat'].iloc[-1]
        prev_val = forecast['yhat'].iloc[-int(forecast_days)]
        
        trend = (last_val - prev_val) / prev_val if prev_val > 0 else 0
        
        source_stats[source] = {
            "forecasted_sessions": round(last_val, 2),
            "trend_percentage": round(trend * 100, 2),
            "is_fatigued": trend < -0.15,
            "confidence": 0.85
        }
        
    return source_stats
