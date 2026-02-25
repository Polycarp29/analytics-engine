import os
import json
import redis
import logging
import requests
from datetime import datetime
import engine
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis Configuration
REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
REDIS_PORT = os.getenv('REDIS_PORT', 6379)
REDIS_DB = os.getenv('REDIS_DB', 0)
REDIS_PREFIX = os.getenv('REDIS_PREFIX', 'metapilot-database-')
JOB_QUEUE = f'{REDIS_PREFIX}analytics:jobs'

# Laravel Webhook Configuration
LARAVEL_URL = os.getenv('APP_URL', 'http://localhost:8000')
WEBHOOK_PATH = '/api/analytics/webhook'

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

def process_job(job_data):
    """
    Processes the analytics job using shared engine logic.
    """
    property_id = job_data.get('property_id')
    job_type = job_data.get('type', 'full') # 'full' or 'ad_performance'
    
    logger.info(f"Processing {job_type} analytics for property: {property_id}")
    
    results = {}
    
    try:
        if job_type == 'ad_performance':
            campaign_data = job_data.get('campaign_data', [])
            keyword_trends = job_data.get('keyword_trends', [])
            recommendations = engine.optimize_budget(campaign_data)
            
            # Interlink with keyword trends
            for rec in recommendations:
                campaign_name = rec['campaign'].lower()
                correlated = [t for t in keyword_trends if t['keyword'].lower() in campaign_name]
                if correlated:
                    top_trend = max(correlated, key=lambda x: x['trend_score'])
                    if top_trend['trend_score'] > 20:
                        rec['action'] = "Scale Up (High Intent)"
            
            results = {
                "property_id": property_id,
                "type": "ad_performance",
                "recommendations": recommendations
            }
        else:
            historical_data = job_data.get('historical_data', [])
            config = job_data.get('config', {})
            
            if not historical_data:
                logger.error("No historical data provided")
                return

            latest = historical_data[-1]
            propensity = engine.calculate_propensity_score(
                latest['channels'],
                latest['returning_users'],
                latest['sessions']
            )
            
            fatigue = engine.detect_source_fatigue(
                historical_data,
                int(config.get("forecast_days", 14))
            )
            
            rankings = []
            for channel, prob in propensity.items():
                chan_data = latest['channels'].get(channel, {'conversions': 0, 'users': 0})
                rankings.append({
                    "channel": channel,
                    "propensity": prob,
                    "efficiency_index": round(prob * (chan_data['conversions'] / chan_data['users'] if chan_data['users'] > 0 else 0), 4)
                })
            rankings = sorted(rankings, key=lambda x: x['efficiency_index'], reverse=True)
            
            results = {
                "property_id": property_id,
                "type": "full",
                "predictions": {
                    "propensity_scores": propensity,
                    "source_fatigue": fatigue,
                    "performance_rankings": rankings
                }
            }

        # Send results back to Laravel
        send_webhook(results)
        
    except Exception as e:
        logger.error(f"Error processing analytics: {e}", exc_info=True)

def send_webhook(results):
    """
    Sends the processed data back to Laravel via a secure webhook.
    """
    webhook_url = f"{LARAVEL_URL.rstrip('/')}{WEBHOOK_PATH}"
    logger.info(f"Sending results to {webhook_url}")
    
    try:
        response = requests.post(webhook_url, json=results, timeout=10)
        if response.status_code == 200:
            logger.info("Webhook delivered successfully")
        else:
            logger.warning(f"Webhook delivered with status: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to deliver webhook: {e}")

def main():
    logger.info("Analytics Worker started, listening for jobs...")
    while True:
        try:
            _, message = r.blpop(JOB_QUEUE)
            job_data = json.loads(message)
            process_job(job_data)
        except Exception as e:
            logger.error(f"Worker Loop Error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
