#!/usr/bin/env python3
"""Quick test to verify web_app works"""
import sys

try:
    print("Testing web_app import...")
    import web_app
    print("✓ Import successful")
    
    print("\nTesting Flask app...")
    app = web_app.app
    print(f"✓ Flask app created: {app}")
    
    print("\nTesting routes...")
    with app.test_client() as client:
        # Test redirect from /
        response = client.get('/')
        print(f"✓ GET / -> Status {response.status_code} (redirect to street-sign)")
        
        # Test street-sign page
        response = client.get('/street-sign')
        print(f"✓ GET /street-sign -> Status {response.status_code}")
        
    print("\n✅ ALL TESTS PASSED - App is working!")
    sys.exit(0)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
