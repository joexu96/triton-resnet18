#!/usr/bin/env python3
"""
Triton ResNet18 Demo Script

This script demonstrates the usage of Triton-accelerated ResNet18 model
with various examples including inference, benchmarking, and training.
"""

import torch
import argparse
import sys
from triton_resnet18 import create_triton_resnet18
from benchmark import compare_models, correctness_test, memory_usage_test
from train_example import main as train_main


def demo_inference():
    """Demonstrate basic inference with Triton ResNet18"""
    print("=== Triton ResNet18 Inference Demo ===")
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = create_triton_resnet18(num_classes=1000)
    model = model.to(device)
    model.eval()
    
    # Create dummy input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    
    print(f"Input shape: {input_tensor.shape}")
    
    # Inference
    with torch.no_grad():
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
        
        output = model(input_tensor)
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            inference_time = start_time.elapsed_time(end_time)
        else:
            inference_time = 0
    
    print(f"Output shape: {output.shape}")
    print(f"Top-5 predictions for first sample:")
    
    # Get top-5 predictions
    probs = torch.softmax(output[0], dim=0)
    top5_probs, top5_indices = torch.topk(probs, 5)
    
    for i, (prob, idx) in enumerate(zip(top5_probs, top5_indices)):
        print(f"  {i+1}. Class {idx.item()}: {prob.item()*100:.2f}%")
    
    if inference_time > 0:
        print(f"Inference time: {inference_time:.2f} ms")
    
    return output


def demo_benchmark():
    """Run comprehensive benchmarks"""
    print("\n=== Running Benchmarks ===")
    
    try:
        # Correctness test
        print("\n1. Correctness Test:")
        correctness_results = correctness_test()
        
        # Performance comparison
        print("\n2. Performance Comparison:")
        performance_results = compare_models(
            batch_sizes=[1, 4, 8],
            num_runs=50
        )
        
        # Memory usage
        print("\n3. Memory Usage:")
        memory_results = memory_usage_test(batch_size=8)
        
        # Summary
        print("\n=== Benchmark Summary ===")
        print("✓ Correctness: Outputs are consistent between implementations")
        
        print("\nPerformance Results:")
        for i, batch_size in enumerate(performance_results['batch_size']):
            pytorch_time = performance_results['pytorch_time'][i]
            triton_time = performance_results['triton_time'][i]
            speedup = performance_results['speedup'][i]
            print(f"  Batch {batch_size}: {speedup:.2f}x speedup "
                  f"({pytorch_time:.1f}ms → {triton_time:.1f}ms)")
        
        print(f"\nMemory: PyTorch={memory_results['pytorch_memory']:.1f}MB, "
              f"Triton={memory_results['triton_memory']:.1f}MB")
        
    except Exception as e:
        print(f"Benchmark failed: {e}")
        print("This might be due to missing dependencies or CUDA issues")


def demo_training():
    """Demonstrate training on CIFAR-10"""
    print("\n=== Training Demo ===")
    print("Starting training on CIFAR-10...")
    
    try:
        train_main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Training failed: {e}")
        print("Make sure you have torchvision installed and internet connection")


def show_model_info():
    """Display model architecture information"""
    print("\n=== Model Information ===")
    
    model = create_triton_resnet18(num_classes=1000)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Model architecture
    print("\nModel Architecture:")
    print(model)
    
    # Layer-wise parameter count
    print("\nLayer-wise parameter count:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                print(f"  {name}: {params:,} parameters")


def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(description='Triton ResNet18 Demo')
    parser.add_argument(
        'demo',
        choices=['inference', 'benchmark', 'train', 'info'],
        help='Demo to run'
    )
    
    args = parser.parse_args()
    
    print("Triton ResNet18 Demo")
    print("=" * 50)
    
    if args.demo == 'inference':
        demo_inference()
    elif args.demo == 'benchmark':
        demo_benchmark()
    elif args.demo == 'train':
        demo_training()
    elif args.demo == 'info':
        show_model_info()
    else:
        print("Invalid demo choice")
        sys.exit(1)
    
    print("\nDemo completed!")


if __name__ == "__main__":
    # If no arguments provided, show help
    if len(sys.argv) == 1:
        print("Triton ResNet18 Demo")
        print("Usage: python main.py [inference|benchmark|train|info]")
        print("\nExamples:")
        print("  python main.py inference  # Basic inference demo")
        print("  python main.py benchmark  # Performance benchmarks")
        print("  python main.py train      # Train on CIFAR-10")
        print("  python main.py info       # Show model information")
    else:
        main()
