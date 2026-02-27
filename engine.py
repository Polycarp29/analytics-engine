import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
from prophet import Prophet
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

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

    # 5. Compute Propensity Scores per channel using the latest snapshot
    latest = history[-1] if history else {}
    channels_data = latest.get('channels', {})
    returning_users = latest.get('returning_users', 0)
    sessions = latest.get('sessions', 1)  # avoid div-by-zero
    propensity_scores = calculate_propensity_score(channels_data, returning_users, sessions)

    # 6. Compute Performance Rankings across all channels
    performance_rankings = _compute_performance_rankings(history)

    return {
        "summary": summary,
        "forecast": forecasts,
        "recommendations": recommendations,
        "propensity_scores": propensity_scores,
        "performance_rankings": performance_rankings,
        "diagnostics": {
            "data_points": len(history),
            "ads_campaigns": len(ads_data),
            "gsc_queries": len(gsc_queries)
        }
    }

def _forecast_metrics(history: List[Dict], days: int) -> Dict:
    """
    Predicts sessions and conversions up to 180 days out.
    Uses date-based lookup to find the correct forecast row for each horizon.
    """
    forecast_results = {}
    metrics = ['sessions', 'conversions']

    for metric in metrics:
        df = pd.DataFrame([
            {"ds": h['date'], "y": h.get(metric, 0)}
            for h in history
        ])

        if len(df) < 14:
            logger.warning(f"Not enough data to forecast '{metric}' — need at least 14 rows, got {len(df)}.")
            continue

        try:
            m = Prophet(interval_width=0.8, yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
            m.fit(df)

            future = m.make_future_dataframe(periods=days)
            forecast = m.predict(future)
            forecast['ds'] = pd.to_datetime(forecast['ds'])

            today = pd.Timestamp.now().normalize()
            periods = [30, 90, 180]
            metric_forecast = {}

            for p in periods:
                target_date = today + timedelta(days=p)
                # Find the closest available date in the forecast
                idx = (forecast['ds'] - target_date).abs().idxmin()
                row = forecast.loc[idx]
                metric_forecast[f"{p}d"] = {
                    "predicted": round(float(row['yhat']), 2),
                    "lower": round(float(row['yhat_lower']), 2),
                    "upper": round(float(row['yhat_upper']), 2),
                    "trend": "up" if row['trend'] > forecast['trend'].iloc[0] else "down"
                }

            forecast_results[metric] = metric_forecast

        except Exception as e:
            logger.error(f"Prophet failed to forecast metric '{metric}': {e}", exc_info=True)

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

    for q in queries[:20]:  # Top 20 organic queries
        q_name = q.get('name', '').lower()
        if not q_name:
            continue

        clicks = q.get('clicks', 0)
        if clicks > 50 and q_name not in ad_keywords:
            insights.append({
                "type": "opportunity_gap",
                "label": f"High intent query '{q_name}' has no ad coverage",
                "keyword": q_name,  # Dedicated field for safe extraction
                "metric": "clicks",
                "value": clicks
            })

    # B. UX Friction (High Bounce Rate on Top Pages)
    latest_history = history[-7:]  # Look at last week
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
        city_name = top_city.get('name', 'Unknown')
        if top_city.get('activeUsers', 0) > 0:
            insights.append({
                "type": "geo_focus",
                "label": f"Primary growth opportunity in {city_name}",
                "city": city_name,  # Dedicated field for safe extraction
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
            # Use the dedicated 'keyword' field instead of parsing the label string
            keyword = insight.get('keyword', 'query')
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

    # Geo Recommendation — use dedicated 'city' field instead of parsing label
    geo_insight = next((i for i in correlations if i['type'] == 'geo_focus'), None)
    if geo_insight:
        city_name = geo_insight.get('city', geo_insight['label'])
        recs.append({
            "type": "ads",
            "priority": "medium",
            "title": f"Expand Targeting in {city_name}",
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
    if df.empty:
        return []
    agg = df.groupby('campaign_name').agg({'cost': 'sum', 'conversions': 'sum'}).reset_index()
    agg['roas'] = agg['conversions'] / agg['cost'].replace(0, np.inf)
    return [
        {
            "campaign": r['campaign_name'],
            "current_spend": float(r['cost']),
            "roas": round(float(r['roas']), 2) if not np.isinf(r['roas']) else 0.0,
            "action": "Scale" if r['roas'] > 1 else "Optimize",
            "reason": f"Campaign shows strong performance with {round(float(r['roas']), 2)}x ROAS." if r['roas'] > 1 else f"ROI is currently below target at {round(float(r['roas']), 2)}x. Optimization recommended."
        }
        for _, r in agg.iterrows()
    ]

def calculate_propensity_score(channels_data: Dict[str, Dict], returning_users: int, sessions: int) -> Dict[str, float]:
    """
    Estimates conversion propensity per channel.
    Formula: weight returning-user loyalty + channel conversion share.
    Result is normalised to [0, 1].
    """
    if not channels_data:
        return {}

    total_channel_conversions = sum(
        v.get('conversions', 0) if isinstance(v, dict) else 0
        for v in channels_data.values()
    )
    total_channel_users = sum(
        v.get('users', 0) if isinstance(v, dict) else 0
        for v in channels_data.values()
    ) or 1

    returning_ratio = min(returning_users / max(sessions, 1), 1.0)

    scores = {}
    for channel, stats in channels_data.items():
        if not isinstance(stats, dict):
            continue
        ch_users = stats.get('users', 0) or 0
        ch_conversions = stats.get('conversions', 0) or 0

        # Conversion rate within the channel
        ch_cvr = ch_conversions / max(ch_users, 1)

        # Share of total conversions for this channel
        conv_share = ch_conversions / max(total_channel_conversions, 1)

        # Weighted score: 50% channel CVR + 30% returning loyalty + 20% conversion share
        raw_score = (0.50 * ch_cvr) + (0.30 * returning_ratio) + (0.20 * conv_share)
        scores[channel] = round(min(raw_score, 1.0), 4)

    return scores

def _compute_performance_rankings(history: List[Dict]) -> List[Dict]:
    """
    Aggregates all channel data across history, computes a propensity score
    and efficiency_index per channel, and returns a sorted ranking list.
    """
    channel_totals: Dict[str, Dict] = {}

    for entry in history:
        channels = entry.get('channels', {})
        sessions = entry.get('sessions', 1) or 1
        returning = entry.get('returning_users', 0)

        for ch_name, stats in channels.items():
            if not isinstance(stats, dict):
                continue
            if ch_name not in channel_totals:
                channel_totals[ch_name] = {'users': 0, 'conversions': 0, 'sessions_sum': 0, 'returning_sum': 0, 'days': 0}
            channel_totals[ch_name]['users'] += stats.get('users', 0)
            channel_totals[ch_name]['conversions'] += stats.get('conversions', 0)
            channel_totals[ch_name]['sessions_sum'] += sessions
            channel_totals[ch_name]['returning_sum'] += returning
            channel_totals[ch_name]['days'] += 1

    if not channel_totals:
        return []

    max_users = max(v['users'] for v in channel_totals.values()) or 1
    rankings = []

    for ch_name, totals in channel_totals.items():
        users = totals['users']
        convs = totals['conversions']
        avg_sessions = totals['sessions_sum'] / max(totals['days'], 1)
        avg_returning = totals['returning_sum'] / max(totals['days'], 1)

        # Propensity: blended CVR + loyalty signal
        cvr = convs / max(users, 1)
        loyalty = avg_returning / max(avg_sessions, 1)
        propensity = round(min((0.6 * cvr) + (0.4 * loyalty), 1.0), 4)

        # Efficiency index: how much of total traffic does this channel contribute × conversion rate
        efficiency_index = round((users / max_users) * (1 + cvr), 4)
        efficiency_index = round(min(efficiency_index, 1.0), 4)

        rankings.append({
            'channel': ch_name,
            'propensity': propensity,
            'efficiency_index': efficiency_index,
            'total_users': users,
            'total_conversions': convs,
        })

    # Sort by efficiency_index descending
    rankings.sort(key=lambda x: x['efficiency_index'], reverse=True)
    return rankings

def detect_source_fatigue(history: List[Dict], forecast_days: int) -> Dict[str, Dict]:
    return {}

def predict_keyword_decay(history: List[Dict]) -> Dict:
    """
    Predicts the decay or resurgence of a search keyword trend.
    History is a list of {date, interest_value}.
    """
    if len(history) < 7:
        return {
            "decay_status": "stable",
            "velocity": 0,
            "forecast_30d": 0,
            "resurgence_probability": 0
        }

    # Prepare data for regression
    df = pd.DataFrame(history)
    df['ds'] = pd.to_datetime(df['date'])
    df['y'] = df['interest_value'].astype(float)
    df = df.sort_values('ds')

    # Calculate Velocity (7-day weighted change)
    recent = df.tail(7)
    if len(recent) >= 2:
        velocity = (recent['y'].iloc[-1] - recent['y'].iloc[0]) / len(recent)
    else:
        velocity = 0

    # Simple Linear Regression for forecast (x is days from first record)
    df['x'] = (df['ds'] - df['ds'].min()).dt.days
    x = df['x'].values
    y = df['y'].values
    
    # Fit line: y = mx + c
    if len(x) > 1:
        m, c = np.polyfit(x, y, 1)
    else:
        m, c = 0, y[0]

    # Forecast target points
    last_x = x[-1]
    forecast_30 = max(0, m * (last_x + 30) + c)
    forecast_90 = max(0, m * (last_x + 90) + c)

    # Determine status
    status = "stable"
    if velocity > 5:
        status = "rising"
    elif velocity < -3:
        status = "decaying"
    
    if df['y'].max() > 70 and df['y'].iloc[-1] < 20:
        status = "dormant"
    
    # Resurgence probability (if dormant but velocity starts increasing)
    resurgence_prob = 0
    if status == "dormant" and velocity > 1:
        resurgence_prob = min(0.9, velocity / 10)
    elif status == "dormant":
        resurgence_prob = 0.1

    return {
        "decay_status": status,
        "velocity": round(float(velocity), 2),
        "forecast_30d": round(float(forecast_30), 2),
        "forecast_90d": round(float(forecast_90), 2),
        "resurgence_probability": round(float(resurgence_prob), 2),
        "slope": round(float(m), 4)
    }
