import global_trends_crawler
import json

def test_crawler():
    print("Testing Global Trends Crawler...")
    
    # Test US daily trends
    print("\nFetching US Daily Trends...")
    us_trends = global_trends_crawler.discover_global_trends(geo='US')
    print(f"Discovered {len(us_trends)} trends for US")
    if us_trends:
        print(f"Sample: {us_trends[0]}")
    
    # Test KE daily trends
    print("\nFetching KE Daily Trends...")
    ke_trends = global_trends_crawler.discover_global_trends(geo='KE', niches=['Real Estate', 'Betting', 'Casino'])
    print(f"Discovered {len(ke_trends)} trends for KE")
    if ke_trends:
        print(f"Sample: {ke_trends[0]}")

if __name__ == "__main__":
    test_crawler()
