import os
import json
import redis
import logging
import requests
from datetime import datetime
import traceback
import engine
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis Configuration — cast to int to avoid type errors when env vars are strings
REDIS_HOST = os.getenv('REDIS_HOST', '127.0.0.1')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))
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
    job_type = job_data.get('type', 'full')  # 'full' or 'ad_performance'

    logger.info(f"Processing {job_type} analytics for property: {property_id}")

    results = None
    try:
        if job_type == 'ad_performance':
            # Specific ad performance optimization
            campaign_data = job_data.get('campaign_data', [])
            recommendations = engine.optimize_budget(campaign_data)
            results = {
                "property_id": property_id,
                "type": "ad_performance",
                "recommendations": recommendations
            }
        else:
            # Full Strategic Analysis (Correlates GA4 + GSC + Ads)
            results = engine.generate_strategic_analysis(job_data)
            results['type'] = 'full'
            results['property_id'] = property_id

        # Send results back to Laravel
        send_webhook(results)

    except Exception as e:
        logger.error(f"Error processing analytics for property {property_id}: {e}")
        logger.error(traceback.format_exc())
        # Only attempt webhook if we have partial results to report
        if results is not None:
            send_webhook(results)

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
