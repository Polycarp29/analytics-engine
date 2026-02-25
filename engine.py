import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from prophet import Prophet
from typing import List, Dict, Optional

def generate_strategic_analysis(data: Dict) -> Dict:
    """
    Main entry point for multi-source correlation, forecasting, and strategy.
    """
    history = data.get('historical_data', [])
    ads_data = data.get('google_ads_data', [])
    gsc_queries = data.get('top_queries', [])
    config = data.get('config', {})
    forecast_days = int(config.get('forecast_days', 90))

    # 1. Forecast Core Metrics
    forecasts = _forecast_metrics(history, forecast_days)

    # 2. Identify Correlations & Anomalies
    correlations = _identify_correlations(data)

    # 3. Generate Strategic Recommendations
    recommendations = _generate_recommendations(data, forecasts, correlations)

    # 4. Generate Summary
    summary = _generate_summary(correlations, forecasts)

    return {
        "summary": summary,
        "forecast": forecasts,
        "recommendations": recommendations,
        "diagnostics": {
            "data_points": len(history),
            "ads_campaigns": len(ads_data),
            "gsc_queries": len(gsc_queries)
        }
    }

def _forecast_metrics(history: List[Dict], days: int) -> Dict:
    """
    Predicts sessions and conversions up to 180 days out.
    """
    forecast_results = {}
    metrics = ['sessions', 'conversions']
    
    for metric in metrics:
        df = pd.DataFrame([
            {"ds": h['date'], "y": h.get(metric, 0)}
            for h in history
        ])
        
        if len(df) < 14:
            continue
            
        try:
            m = Prophet(interval_width=0.8, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            m.fit(df)
            
            future = m.make_future_dataframe(periods=days)
            forecast = m.predict(future)
            
            # Extract point predictions for 30, 90, 180 days
            periods = [30, 90, 180]
            metric_forecast = {}
            
            for p in periods:
                idx = -1 if p >= days else -(days - p)
                if abs(idx) > len(forecast):
                    continue
                    
                row = forecast.iloc[idx]
                metric_forecast[f"{p}d"] = {
                    "predicted": round(float(row['yhat']), 2),
                    "lower": round(float(row['yhat_lower']), 2),
                    "upper": round(float(row['yhat_upper']), 2),
                    "trend": "up" if row['trend'] > forecast['trend'].iloc[0] else "down"
                }
            
            forecast_results[metric] = metric_forecast
        except Exception as e:
            # Silent fail for individual metrics in production
            pass
            
    return forecast_results

def _identify_correlations(data: Dict) -> List[Dict]:
    """
    Analyses interactions between GSC, Ads, and GA4.
    """
    insights = []
    history = data.get('historical_data', [])
    ads = data.get('google_ads_data', [])
    queries = data.get('top_queries', [])
    
    # A. Paid vs Organic Gap (Missing Ad Coverage)
    ad_keywords = set()
    for campaign in ads:
        for kw in campaign.get('keywords', []):
            if isinstance(kw, str):
                ad_keywords.add(kw.lower())
    
    for q in queries[:20]: # Top 20 organic queries
        q_name = q.get('name', '').lower()
        if not q_name: continue
        
        clicks = q.get('clicks', 0)
        if clicks > 50 and q_name not in ad_keywords:
            insights.append({
                "type": "opportunity_gap",
                "label": f"High intent query '{q_name}' has no ad coverage",
                "metric": "clicks",
                "value": clicks
            })

    # B. UX Friction (High Bounce Rate on Top Pages)
    latest_history = history[-7:] # Look at last week
    if latest_history:
        avg_bounce = np.mean([h.get('bounce_rate', 0) for h in latest_history])
        if avg_bounce > 75:
            insights.append({
                "type": "ux_friction",
                "label": "High site-wide bounce rate detected",
                "value": f"{avg_bounce:.1f}%"
            })

    # C. Geo Imbalance
    cities = data.get('by_city', [])
    if cities:
        top_city = max(cities, key=lambda x: x.get('activeUsers', 0))
        if top_city.get('activeUsers', 0) > 0:
            insights.append({
                "type": "geo_focus",
                "label": f"Primary growth opportunity in {top_city['name']}",
                "value": top_city['activeUsers']
            })

    return insights

def _generate_recommendations(data: Dict, forecasts: Dict, correlations: List[Dict]) -> List[Dict]:
    """
    Builds prioritized action list.
    """
    recs = []
    
    for insight in correlations:
        if insight['type'] == 'opportunity_gap':
            keyword = insight['label'].split("'")[1] if "'" in insight['label'] else "query"
            recs.append({
                "type": "ads",
                "priority": "high",
                "title": f"Create Ad Group for '{keyword}'",
                "rationale": f"This keyword drives {insight['value']} organic clicks but has 0 paid visibility.",
                "expected_impact": {"conversions": "+5-10/mo", "roi_increase": "15%"},
                "confidence": 0.85,
                "actions": [{"step": "setup", "description": "Create a Search Campaign or Ad Group targeting this exact keyword."}]
            })
        
        elif insight['type'] == 'ux_friction':
            recs.append({
                "type": "content",
                "priority": "critical",
                "title": "Optimize Landing Page UX",
                "rationale": f"Bounce rate is critical at {insight['value']}. This suppresses conversion efficiency.",
                "expected_impact": {"conversion_rate": "+15%"},
                "confidence": 0.92,
                "actions": [
                    {"step": "audit", "description": "Check PageSpeed Insights for LCP and CLS issues."},
                    {"step": "ux", "description": "Ensure 'Above the Fold' content matches user intent."}
                ]
            })

    # Add Default Growth Recs based on Forecasts
    conv_forecast = forecasts.get('conversions', {}).get('90d', {})
    if conv_forecast.get('trend') == 'down':
        recs.append({
            "type": "seo",
            "priority": "high",
            "title": "Refresh Aging Content Assets",
            "rationale": "90-day conversion forecast shows a downward trend. Top pages may be losing relevance.",
            "expected_impact": {"retention": "Stop the decline"},
            "confidence": 0.78,
            "actions": [{"step": "update", "description": "Identify pages with >20% traffic drop and update with fresh data/images."}]
        })

    # Geo Recommendation
    geo_insight = next((i for i in correlations if i['type'] == 'geo_focus'), None)
    if geo_insight:
        recs.append({
            "type": "ads",
            "priority": "medium",
            "title": f"Expand Targeting in {geo_insight['label'].split('in ')[1]}",
            "rationale": f"High organic traction in this region ({geo_insight['value']} users) suggests strong local demand.",
            "expected_impact": {"sessions": "+15-20%"},
            "confidence": 0.8,
            "actions": [{"step": "geo", "description": "Shift budget to specific city-level targets in this region."}]
        })

    return recs

def _generate_summary(correlations: List[Dict], forecasts: Dict) -> str:
    """
    Synthesizes a short natural language overview.
    """
    if not correlations:
        return "Performance is stable. Continue monitoring current trends."
        
    main_issue = correlations[0]['label']
    trend_msg = ""
    if 'sessions' in forecasts:
        trend = forecasts['sessions'].get('90d', {}).get('trend', 'stable')
        trend_msg = f" Traffic is forecasted to trend {trend} over the next 3 months."
        
    return f"Performance Update: {main_issue}.{trend_msg}"

# Legacy functions for compatibility if needed, though generate_strategic_analysis is preferred
def optimize_budget(campaign_data: List[Dict]) -> List[Dict]:
    df = pd.DataFrame(campaign_data)
    if df.empty: return []
    agg = df.groupby('campaign_name').agg({'cost': 'sum', 'conversions': 'sum'}).reset_index()
    agg['roas'] = agg['conversions'] / agg['cost'].replace(0, np.inf)
    return [{"campaign": r['campaign_name'], "action": "Scale" if r['roas'] > 1 else "Optimize"} for _, r in agg.iterrows()]

def calculate_propensity_score(channels_data: Dict[str, Dict], returning_users: int, sessions: int) -> Dict[str, float]:
    return {c: 0.5 for c in channels_data.keys()}

def detect_source_fatigue(history: List[Dict], forecast_days: int) -> Dict[str, Dict]:
    return {}
