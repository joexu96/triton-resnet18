#!/usr/bin/env python3
"""
Test script to verify Triton ResNet18 installation
"""

import sys
import torch
import warnings

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import triton
        print("✓ Triton imported successfully")
    except ImportError as e:
        print(f"✗ Triton import failed: {e}")
        return False
    
    try:
        from triton_resnet18 import create_triton_resnet18
        print("✓ Triton ResNet18 imported successfully")
    except ImportError as e:
        print(f"✗ Triton ResNet18 import failed: {e}")
        return False
    
    return True

def test_model_creation():
    """Test model creation and basic functionality"""
    print("\nTesting model creation...")
    
    try:
        from triton_resnet18 import create_triton_resnet18
        
        # Create model
        model = create_triton_resnet18(num_classes=10)
        print("✓ Model created successfully")
        
        # Test forward pass
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Create dummy input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(input_tensor)
        
        expected_shape = (batch_size, 10)
        if output.shape == expected_shape:
            print(f"✓ Forward pass successful, output shape: {output.shape}")
        else:
            print(f"✗ Unexpected output shape: {output.shape}, expected: {expected_shape}")
            return False
            
    except Exception as e:
        print(f"✗ Model creation/forward pass failed: {e}")
        return False
    
    return True

def test_device_compatibility():
    """Test device compatibility"""
    print("\nTesting device compatibility...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("Running on CPU (CUDA not available)")
    
    return True

def test_triton_compilation():
    """Test Triton kernel compilation"""
    print("\nTesting Triton kernel compilation...")
    
    try:
        from triton_resnet18 import TritonConv2d
        
        # Create a small Triton layer
        layer = TritonConv2d(3, 16, kernel_size=3, padding=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        layer = layer.to(device)
        
        # Test with small input
        input_tensor = torch.randn(1, 3, 32, 32).to(device)
        
        layer.eval()
        with torch.no_grad():
            output = layer(input_tensor)
        
        expected_shape = (1, 16, 32, 32)
        if output.shape == expected_shape:
            print("✓ Triton kernel compilation successful")
        else:
            print(f"✗ Triton kernel output shape mismatch: {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Triton kernel compilation failed: {e}")
        # This might be expected on CPU
        if not torch.cuda.is_available():
            print("  (This is expected when running on CPU)")
            return True
        return False
    
    return True

def main():
    """Run all tests"""
    print("Triton ResNet18 Installation Test")
    print("=" * 40)
    
    tests = [
        ("Import tests", test_imports),
        ("Model creation", test_model_creation),
        ("Device compatibility", test_device_compatibility),
        ("Triton compilation", test_triton_compilation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "=" * 40)
    print("Test Summary:")
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✓ PASS" if results[i] else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Triton ResNet18 is ready to use.")
        return 0
    else:
        print("\n❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())