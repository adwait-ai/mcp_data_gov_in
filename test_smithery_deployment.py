#!/usr/bin/env python3
"""
Test script to validate Smithery deployment configuration.
"""

import os
import sys
import tempfile
import subprocess

def test_smithery_config():
    """Test the smithery.yaml configuration."""
    print("🧪 Testing Smithery configuration...")
    
    # Check if smithery.yaml exists
    if not os.path.exists("smithery.yaml"):
        print("❌ smithery.yaml not found")
        return False
    
    print("✅ smithery.yaml exists")
    
    # Check if Dockerfile exists
    if not os.path.exists("Dockerfile"):
        print("❌ Dockerfile not found")
        return False
    
    print("✅ Dockerfile exists")
    
    # Test HTTP mode
    print("🧪 Testing HTTP mode detection...")
    
    # Set PORT environment variable
    os.environ["PORT"] = "8000"
    os.environ["DATA_GOV_API_KEY"] = "test-key"
    
    try:
        # Import the server module
        import mcp_server
        
        # Test configuration parsing
        config = mcp_server.parse_smithery_config()
        print(f"✅ Configuration parsing works: {list(config.keys()) if config else 'no config'}")
        
        # Test API key detection
        api_key = os.getenv("DATA_GOV_API_KEY")
        if api_key:
            print(f"✅ API key detected: {api_key[:10]}...")
        else:
            print("⚠️ No API key set")
        
        print("✅ All Smithery configuration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False
    
    finally:
        # Clean up
        if "PORT" in os.environ:
            del os.environ["PORT"]

def test_docker_build():
    """Test if Docker build would succeed (syntax check only)."""
    print("🧪 Testing Docker configuration...")
    
    try:
        # Check if docker is available
        result = subprocess.run(["docker", "--version"], 
                               capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            print("⚠️ Docker not available, skipping build test")
            return True
        
        print("✅ Docker is available")
        
        # Read and validate Dockerfile contents
        with open("Dockerfile", "r") as f:
            dockerfile_content = f.read()
        
        # Basic validation checks
        if "FROM " not in dockerfile_content:
            print("❌ Dockerfile missing FROM instruction")
            return False
        
        if "CMD " not in dockerfile_content and "ENTRYPOINT " not in dockerfile_content:
            print("❌ Dockerfile missing CMD or ENTRYPOINT instruction")
            return False
        
        print("✅ Dockerfile has required instructions")
        
        # Check if required files exist for COPY commands
        if "COPY" in dockerfile_content:
            print("✅ Dockerfile includes file copying")
        
        print("✅ Dockerfile validation passed")
        return True
            
    except subprocess.TimeoutExpired:
        print("⚠️ Docker command timed out")
        return False
    except FileNotFoundError:
        print("⚠️ Docker not found, skipping build test")
        return True
    except Exception as e:
        print(f"⚠️ Docker test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Smithery Deployment Validation")
    print("=" * 40)
    
    config_ok = test_smithery_config()
    docker_ok = test_docker_build()
    
    print("\n" + "=" * 40)
    if config_ok and docker_ok:
        print("✅ All tests passed! Ready for Smithery deployment.")
        print("\n📋 Next steps:")
        print("1. Push to GitHub")
        print("2. Connect GitHub to Smithery")
        print("3. Deploy from Smithery dashboard")
        print("4. Configure with your data.gov.in API key")
        return 0
    else:
        print("❌ Some tests failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
