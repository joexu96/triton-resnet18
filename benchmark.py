import torch
import torch.nn as nn
import torchvision.models as models
import time
import numpy as np
from triton_resnet18 import TritonResNet18, create_triton_resnet18, benchmark_model


def create_pytorch_resnet18(num_classes: int = 1000):
    """Create standard PyTorch ResNet18"""
    model = models.resnet18(pretrained=False)
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compare_models(
    batch_sizes=[1, 4, 8, 16],
    input_size=(224, 224),
    num_classes=1000,
    num_runs=100
):
    """Compare Triton and PyTorch ResNet18 performance"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    results = {
        'batch_size': [],
        'pytorch_time': [],
        'triton_time': [],
        'speedup': []
    }
    
    for batch_size in batch_sizes:
        print(f"\nTesting batch size: {batch_size}")
        
        # Create models
        pytorch_model = create_pytorch_resnet18(num_classes).to(device)
        triton_model = create_triton_resnet18(num_classes).to(device)
        
        # Create input
        input_tensor = torch.randn(batch_size, 3, *input_size).to(device)
        
        # Benchmark PyTorch model
        pytorch_time = benchmark_model(pytorch_model, input_tensor, num_runs)
        print(f"PyTorch ResNet18: {pytorch_time*1000:.2f} ms")
        
        # Benchmark Triton model
        triton_time = benchmark_model(triton_model, input_tensor, num_runs)
        print(f"Triton ResNet18: {triton_time*1000:.2f} ms")
        
        # Calculate speedup
        speedup = pytorch_time / triton_time if triton_time > 0 else 1.0
        print(f"Speedup: {speedup:.2f}x")
        
        # Store results
        results['batch_size'].append(batch_size)
        results['pytorch_time'].append(pytorch_time * 1000)  # Convert to ms
        results['triton_time'].append(triton_time * 1000)
        results['speedup'].append(speedup)
    
    return results


def memory_usage_test(batch_size=8, input_size=(224, 224)):
    """Test memory usage of both models"""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping memory test")
        return
    
    device = torch.device("cuda")
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # PyTorch model
    pytorch_model = create_pytorch_resnet18().to(device)
    input_tensor = torch.randn(batch_size, 3, *input_size).to(device)
    
    with torch.no_grad():
        _ = pytorch_model(input_tensor)
    
    pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Triton model
    triton_model = create_triton_resnet18().to(device)
    
    with torch.no_grad():
        _ = triton_model(input_tensor)
    
    triton_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"\nMemory Usage Comparison (batch_size={batch_size}):")
    print(f"PyTorch ResNet18: {pytorch_memory:.2f} MB")
    print(f"Triton ResNet18: {triton_memory:.2f} MB")
    print(f"Memory difference: {abs(triton_memory - pytorch_memory):.2f} MB")
    
    return {
        'pytorch_memory': pytorch_memory,
        'triton_memory': triton_memory
    }


def correctness_test():
    """Test if Triton model produces similar outputs to PyTorch model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create models
    pytorch_model = create_pytorch_resnet18().to(device)
    triton_model = create_triton_resnet18().to(device)
    
    # Set both models to eval mode
    pytorch_model.eval()
    triton_model.eval()
    
    # Create test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    with torch.no_grad():
        pytorch_output = pytorch_model(input_tensor)
        triton_output = triton_model(input_tensor)
    
    # Compare outputs
    diff = torch.abs(pytorch_output - triton_output)
    max_diff = torch.max(diff).item()
    mean_diff = torch.mean(diff).item()
    
    print("\nCorrectness Test Results:")
    print(f"Maximum absolute difference: {max_diff:.6f}")
    print(f"Mean absolute difference: {mean_diff:.6f}")
    
    # Check if outputs are close
    tolerance = 1e-3
    is_close = torch.allclose(pytorch_output, triton_output, atol=tolerance)
    print(f"Outputs are close (atol={tolerance}): {is_close}")
    
    return {
        'max_diff': max_diff,
        'mean_diff': mean_diff,
        'is_close': is_close
    }


def profile_model(model, input_tensor, model_name="Model"):
    """Profile model layer by layer"""
    print(f"\nProfiling {model_name}:")
    
    def print_module_size(module, name):
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        print(f"{name}: {total_params:,} total params, {trainable_params:,} trainable")
    
    print_module_size(model, "Total")
    
    # Profile forward pass
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        output = model(input_tensor)
        end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000
    print(f"Inference time: {inference_time:.2f} ms")
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    print("=== Triton ResNet18 Benchmark Suite ===\n")
    
    # Correctness test
    print("1. Running correctness test...")
    correctness_results = correctness_test()
    
    # Performance comparison
    print("\n2. Running performance comparison...")
    performance_results = compare_models(
        batch_sizes=[1, 4, 8],
        num_runs=50
    )
    
    # Memory usage test
    print("\n3. Running memory usage test...")
    memory_results = memory_usage_test(batch_size=8)
    
    # Summary
    print("\n=== Summary ===")
    print("Performance Results:")
    for i, batch_size in enumerate(performance_results['batch_size']):
        print(f"  Batch {batch_size}: "
              f"PyTorch={performance_results['pytorch_time'][i]:.2f}ms, "
              f"Triton={performance_results['triton_time'][i]:.2f}ms, "
              f"Speedup={performance_results['speedup'][i]:.2f}x")
    
    print(f"\nCorrectness: {'✓' if correctness_results['is_close'] else '✗'}")
    print(f"Memory Usage: PyTorch={memory_results['pytorch_memory']:.1f}MB, "
          f"Triton={memory_results['triton_memory']:.1f}MB")