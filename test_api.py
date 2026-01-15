"""
Simple test script for the Portfolio Risk Intelligence API.

Usage:
    python test_api.py

Make sure the API is running on http://localhost:8000
"""

import requests
import json
from datetime import datetime

API_BASE_URL = "http://localhost:8000/api/v1"

def test_analyze_portfolio():
    """Test the portfolio analysis endpoint."""
    print("=" * 60)
    print("Testing Portfolio Analysis Endpoint")
    print("=" * 60)
    
    # Sample portfolio
    payload = {
        "tickers": ["AAPL", "GOOGL", "MSFT"],
        "weights": [0.4, 0.3, 0.3],
        "analysis_date": datetime.now().isoformat()
    }
    
    print(f"\nRequest payload:")
    print(json.dumps(payload, indent=2))
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/portfolio/analyze",
            json=payload,
            timeout=60  # Analysis may take time
        )
        
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 201:
            result = response.json()
            print(f"\n✅ Analysis successful!")
            print(f"Portfolio ID: {result.get('portfolio_id')}")
            print(f"\nRisk Metrics:")
            risk_metrics = result.get('risk_metrics', {})
            print(f"  - VaR (95%): {risk_metrics.get('var_95', 0):.4f}")
            print(f"  - CVaR (95%): {risk_metrics.get('cvar_95', 0):.4f}")
            print(f"  - Sharpe Ratio: {risk_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  - Max Drawdown: {risk_metrics.get('max_drawdown', 0):.2f}%")
            
            print(f"\nStress Test Results:")
            stress_results = result.get('stress_test_results', {})
            for scenario, data in stress_results.items():
                if isinstance(data, dict):
                    print(f"  - {scenario}: {data.get('peak_drawdown', 0):.2f}% drawdown")
            
            print(f"\nRisk Attributions:")
            attributions = result.get('risk_attributions', [])
            for attr in attributions[:3]:  # Show first 3
                print(f"  - {attr.get('ticker')}: {attr.get('risk_contribution', 0):.1f}% contribution")
            
            return result.get('portfolio_id')
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return None
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Is the API running on http://localhost:8000?")
        return None
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return None


def test_get_report(portfolio_id: str):
    """Test the get report endpoint."""
    if not portfolio_id:
        print("\nSkipping get report test (no portfolio_id)")
        return
    
    print("\n" + "=" * 60)
    print("Testing Get Report Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE_URL}/portfolio/report/{portfolio_id}")
        
        print(f"\nResponse Status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Report retrieved successfully!")
            result = response.json()
            print(f"Portfolio: {', '.join(result.get('portfolio', {}).get('tickers', []))}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")


def test_health_check():
    """Test the health check endpoint."""
    print("=" * 60)
    print("Testing Health Check")
    print("=" * 60)
    
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"\nResponse Status: {response.status_code}")
        if response.status_code == 200:
            print("✅ API is healthy!")
            print(f"Response: {response.json()}")
        else:
            print(f"❌ Health check failed")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    print("\n🧪 Portfolio Risk Intelligence API Test Suite\n")
    
    # Test health check
    test_health_check()
    
    # Test portfolio analysis
    portfolio_id = test_analyze_portfolio()
    
    # Test get report
    test_get_report(portfolio_id)
    
    print("\n" + "=" * 60)
    print("Test Suite Complete")
    print("=" * 60 + "\n")
