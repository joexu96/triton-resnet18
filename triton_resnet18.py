import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Tuple
import math


@triton.jit
def conv2d_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    batch_size, in_channels, out_channels, height, width,
    kernel_size, stride, padding,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for 2D convolution"""
    pid = tl.program_id(0)
    
    # Calculate output position
    out_h = (height + 2 * padding - kernel_size) // stride + 1
    out_w = (width + 2 * padding - kernel_size) // stride + 1
    
    # Calculate batch, channel, and spatial indices
    b = pid // (out_channels * out_h * out_w)
    remaining = pid % (out_channels * out_h * out_w)
    c_out = remaining // (out_h * out_w)
    remaining = remaining % (out_h * out_w)
    h_out = remaining // out_w
    w_out = remaining % out_w
    
    # Calculate input position
    h_in = h_out * stride - padding
    w_in = w_out * stride - padding
    
    # Initialize accumulator
    acc = 0.0
    
    # Perform convolution
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            h = h_in + kh
            w = w_in + kw
            
            if (h >= 0) & (h < height) & (w >= 0) & (w < width):
                for c_in in range(in_channels):
                    # Input indexing: [batch, channel, height, width]
                    input_idx = b * in_channels * height * width + c_in * height * width + h * width + w
                    # Weight indexing: [out_channel, in_channel, kernel_h, kernel_w]
                    weight_idx = c_out * in_channels * kernel_size * kernel_size + c_in * kernel_size * kernel_size + kh * kernel_size + kw
                    
                    input_val = tl.load(input_ptr + input_idx)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    acc += input_val * weight_val
    
    # Add bias if provided
    if bias_ptr is not None:
        bias_val = tl.load(bias_ptr + c_out)
        acc += bias_val
    
    # Store output
    output_idx = b * out_channels * out_h * out_w + c_out * out_h * out_w + h_out * out_w + w_out
    tl.store(output_ptr + output_idx, acc)


@triton.jit
def batch_norm_kernel(
    input_ptr, weight_ptr, bias_ptr, running_mean_ptr, running_var_ptr,
    output_ptr, eps, momentum, training,
    num_features, spatial_size,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for batch normalization"""
    pid = tl.program_id(0)
    
    # Calculate indices
    b = pid // (num_features * spatial_size)
    remaining = pid % (num_features * spatial_size)
    c = remaining // spatial_size
    s = remaining % spatial_size
    
    # Calculate indices
    idx = b * num_features * spatial_size + c * spatial_size + s
    
    # Load input
    x = tl.load(input_ptr + idx)
    
    # Load parameters
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)
    running_mean = tl.load(running_mean_ptr + c)
    running_var = tl.load(running_var_ptr + c)
    
    # Normalize
    x_norm = (x - running_mean) / tl.sqrt(running_var + eps)
    
    # Scale and shift
    y = weight * x_norm + bias
    
    # Store output
    tl.store(output_ptr + idx, y)


@triton.jit
def relu_kernel(input_ptr, output_ptr, size, BLOCK_SIZE: tl.constexpr):
    """Triton kernel for ReLU activation"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # Process block
    for i in range(BLOCK_SIZE):
        idx = block_start + i
        if idx < size:
            val = tl.load(input_ptr + idx)
            val = tl.maximum(val, 0.0)
            tl.store(output_ptr + idx, val)


class TritonConv2d(nn.Module):
    """Triton-accelerated 2D convolution layer"""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, in_channels, height, width = x.shape
        
        # Calculate output dimensions
        out_h = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Allocate output tensor
        output = torch.empty(
            batch_size, self.out_channels, out_h, out_w,
            device=x.device, dtype=x.dtype
        )
        
        # Launch Triton kernel
        grid = lambda meta: (batch_size * self.out_channels * out_h * out_w,)
        
        conv2d_kernel[grid](
            x, self.weight, self.bias, output,
            batch_size, in_channels, self.out_channels, height, width,
            self.kernel_size, self.stride, self.padding,
            BLOCK_SIZE=256
        )
        
        return output


class TritonBatchNorm2d(nn.Module):
    """Triton-accelerated batch normalization layer"""
    
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.shape
        spatial_size = height * width
        
        output = torch.empty_like(x)
        
        grid = lambda meta: (batch_size * channels * spatial_size,)
        
        batch_norm_kernel[grid](
            x, self.weight, self.bias, self.running_mean, self.running_var,
            output, self.eps, self.momentum, self.training,
            channels, spatial_size,
            BLOCK_SIZE=256
        )
        
        return output


class TritonReLU(nn.Module):
    """Triton-accelerated ReLU activation"""
    
    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.inplace:
            output = x
        else:
            output = torch.empty_like(x)
        
        size = x.numel()
        grid = lambda meta: ((size + 255) // 256,)
        
        relu_kernel[grid](
            x, output, size,
            BLOCK_SIZE=256
        )
        
        return output


class BasicBlock(nn.Module):
    """Basic block for ResNet18"""
    
    expansion = 1
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = TritonConv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = TritonBatchNorm2d(out_channels)
        self.relu = TritonReLU(inplace=True)
        
        self.conv2 = TritonConv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = TritonBatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                TritonConv2d(
                    in_channels, out_channels * self.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                TritonBatchNorm2d(out_channels * self.expansion)
            )
        
        self.stride = stride
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class TritonResNet18(nn.Module):
    """Triton-accelerated ResNet18 implementation"""
    
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = TritonConv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = TritonBatchNorm2d(64)
        self.relu = TritonReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # ResNet layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, out_channels: int, blocks: int, stride: int):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def create_triton_resnet18(num_classes: int = 1000) -> TritonResNet18:
    """Factory function to create TritonResNet18 model"""
    return TritonResNet18(num_classes=num_classes)


def benchmark_model(model, input_tensor, num_runs: int = 100):
    """Benchmark model performance"""
    model.eval()
    
    # Warmup
    for _ in range(10):
        _ = model(input_tensor)
    
    # Synchronize
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Benchmark
    import time
    start_time = time.time()
    
    for _ in range(num_runs):
        output = model(input_tensor)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time


if __name__ == "__main__":
    # Test the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_triton_resnet18(num_classes=1000).to(device)
    
    # Create dummy input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # Forward pass
    with torch.no_grad():
        output = model(input_tensor)
    
    print(f"Model output shape: {output.shape}")
    print(f"Expected: [{batch_size}, 1000]")
    
    # Benchmark
    if torch.cuda.is_available():
        avg_time = benchmark_model(model, input_tensor)
        print(f"Average inference time: {avg_time*1000:.2f} ms")