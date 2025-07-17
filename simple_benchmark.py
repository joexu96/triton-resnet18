#!/usr/bin/env python3
"""
Simplified benchmark for Triton ResNet18 without torchvision dependency
"""

import torch
import torch.nn as nn
import time
from triton_resnet18 import create_triton_resnet18, benchmark_model


class SimpleResNet18(nn.Module):
    """Simplified PyTorch ResNet18-like model for comparison"""
    
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # ResNet blocks (simplified)
            self._make_layer(64, 64, 2, stride=1),
            self._make_layer(64, 128, 2, stride=2),
            self._make_layer(128, 256, 2, stride=2),
            self._make_layer(256, 512, 2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        
        # First block
        layers.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ))
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            shortcut = nn.Identity()
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            ))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def simple_benchmark():
    """Simple benchmark comparing Triton vs basic PyTorch implementation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping GPU benchmark")
        return
    
    # Create models
    triton_model = create_triton_resnet18(num_classes=1000).to(device)
    pytorch_model = SimpleResNet18(num_classes=1000).to(device)
    
    # Set to eval mode
    triton_model.eval()
    pytorch_model.eval()
    
    # Test configurations
    batch_sizes = [1, 4, 8]
    num_runs = 50
    
    print("\n=== Triton ResNet18 Performance Test ===")
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        # Create input
        input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
        
        # Benchmark Triton model
        triton_time = benchmark_model(triton_model, input_tensor, num_runs)
        print(f"Triton ResNet18: {triton_time*1000:.2f} ms")
        
        # Benchmark PyTorch model
        pytorch_time = benchmark_model(pytorch_model, input_tensor, num_runs)
        print(f"PyTorch ResNet18: {pytorch_time*1000:.2f} ms")
        
        # Calculate speedup
        if pytorch_time > 0 and triton_time > 0:
            speedup = pytorch_time / triton_time
            print(f"Speedup: {speedup:.2f}x")
        else:
            print("Speedup: N/A")
    
    # Memory test
    print("\n=== Memory Usage Test ===")
    batch_size = 8
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Triton model memory
    with torch.no_grad():
        _ = triton_model(input_tensor)
    triton_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # PyTorch model memory
    with torch.no_grad():
        _ = pytorch_model(input_tensor)
    pytorch_memory = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Triton ResNet18 memory: {triton_memory:.1f} MB")
    print(f"PyTorch ResNet18 memory: {pytorch_memory:.1f} MB")
    print(f"Memory difference: {abs(triton_memory - pytorch_memory):.1f} MB")


def correctness_test():
    """Test correctness of Triton implementation"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=== Correctness Test ===")
    
    # Create models
    triton_model = create_triton_resnet18(num_classes=10).to(device)
    pytorch_model = SimpleResNet18(num_classes=10).to(device)
    
    # Set to eval mode
    triton_model.eval()
    pytorch_model.eval()
    
    # Test input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    with torch.no_grad():
        triton_output = triton_model(input_tensor)
        pytorch_output = pytorch_model(input_tensor)
    
    print(f"Triton output shape: {triton_output.shape}")
    print(f"PyTorch output shape: {pytorch_output.shape}")
    
    # Basic sanity check
    if triton_output.shape == pytorch_output.shape:
        print("✓ Output shapes match")
    else:
        print("✗ Output shapes don't match")
    
    # Check for NaN/Inf
    if torch.isnan(triton_output).any():
        print("✗ Triton output contains NaN")
    else:
        print("✓ Triton output is valid")
    
    if torch.isinf(triton_output).any():
        print("✗ Triton output contains Inf")
    else:
        print("✓ Triton output is finite")


if __name__ == "__main__":
    print("Triton ResNet18 Simple Benchmark")
    print("=" * 40)
    
    correctness_test()
    simple_benchmark()
    
    print("\nBenchmark completed!")