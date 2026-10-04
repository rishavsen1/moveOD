#!/usr/bin/env python3
"""
Test script for MOVE-OD Backend API
Tests all endpoints to ensure they're working correctly
"""

import requests
import json
import time
import sys

API_BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test the root endpoint"""
    print("\n🔍 Testing health check endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        print(f"✅ Health check passed: {data}")
        return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_get_states():
    """Test getting states and counties"""
    print("\n🔍 Testing states endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/states")
        response.raise_for_status()
        data = response.json()
        
        if "states" in data and len(data["states"]) > 0:
            state_count = len(data["states"])
            print(f"✅ States endpoint passed: Found {state_count} states")
            
            # Show first state as example
            first_state = data["states"][0]
            print(f"   Example: {first_state['name']} ({first_state['id']}) with {len(first_state['counties'])} counties")
            return True, data["states"]
        else:
            print("❌ States endpoint returned empty data")
            return False, None
    except Exception as e:
        print(f"❌ States endpoint failed: {e}")
        return False, None

def test_process_endpoint(dry_run=True):
    """Test the processing endpoint (dry run by default)"""
    print("\n🔍 Testing process endpoint...")
    
    request_data = {
        "state": "Tennessee",
        "county": "Hamilton",
        "start_date": "2025-03-10",
        "end_date": "2025-03-10",
        "lodes_year": "2022",
        "tiger_shapefile_year": "2024",
        "use_ms_buildings": True,
        "use_safegraph": False,
        "od_option": "Origin and Destination in same County"
    }
    
    if dry_run:
        print(f"   (Dry run - would send: {json.dumps(request_data, indent=2)})")
        print("   Skipping actual processing to avoid long wait time")
        return True, None
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/process",
            json=request_data,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if "job_id" in data:
            print(f"✅ Process endpoint passed: Job ID = {data['job_id']}")
            return True, data["job_id"]
        else:
            print("❌ Process endpoint returned unexpected data")
            return False, None
    except Exception as e:
        print(f"❌ Process endpoint failed: {e}")
        return False, None

def test_job_status(job_id):
    """Test getting job status"""
    if not job_id:
        print("\n⏭️  Skipping job status test (no job ID)")
        return True
    
    print(f"\n🔍 Testing job status endpoint for job {job_id}...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/job/{job_id}")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ Job status endpoint passed:")
        print(f"   Status: {data.get('status')}")
        print(f"   Progress: {data.get('progress', 0) * 100:.1f}%")
        print(f"   Message: {data.get('message')}")
        return True
    except Exception as e:
        print(f"❌ Job status endpoint failed: {e}")
        return False

def test_invalid_job():
    """Test error handling for invalid job ID"""
    print("\n🔍 Testing error handling (invalid job ID)...")
    try:
        response = requests.get(f"{API_BASE_URL}/api/job/invalid-job-id-123")
        
        if response.status_code == 404:
            print("✅ Error handling passed: Returns 404 for invalid job")
            return True
        else:
            print(f"❌ Error handling failed: Expected 404, got {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def run_all_tests(include_processing=False):
    """Run all API tests"""
    print("=" * 60)
    print("🧪 MOVE-OD Backend API Test Suite")
    print("=" * 60)
    
    results = []
    
    # Test 1: Health check
    results.append(("Health Check", test_health_check()))
    
    # Test 2: Get states
    success, states = test_get_states()
    results.append(("Get States", success))
    
    # Test 3: Process endpoint (dry run by default)
    success, job_id = test_process_endpoint(dry_run=not include_processing)
    results.append(("Process Endpoint", success))
    
    # Test 4: Job status (if we have a job)
    if job_id:
        results.append(("Job Status", test_job_status(job_id)))
    
    # Test 5: Error handling
    results.append(("Error Handling", test_invalid_job()))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    # Check if backend is running
    print("Checking if backend is running...")
    try:
        requests.get(f"{API_BASE_URL}/", timeout=2)
        print("✅ Backend is running\n")
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to backend at {API_BASE_URL}")
        print("Please start the backend first:")
        print("  cd backend && python app.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error connecting to backend: {e}")
        sys.exit(1)
    
    # Run tests
    include_processing = "--full" in sys.argv or "-f" in sys.argv
    
    if include_processing:
        print("⚠️  Running FULL tests including actual processing (may take 10-30 minutes)")
        print("Press Ctrl+C within 5 seconds to cancel...\n")
        try:
            time.sleep(5)
        except KeyboardInterrupt:
            print("\n\n❌ Tests cancelled by user")
            sys.exit(0)
    
    exit_code = run_all_tests(include_processing)
    sys.exit(exit_code)
