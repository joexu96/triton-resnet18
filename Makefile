.PHONY: install test benchmark train clean help

# Default target
help:
	@echo "Triton ResNet18 Makefile"
	@echo "======================="
	@echo ""
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run installation tests"
	@echo "  benchmark   - Run performance benchmarks"
	@echo "  train       - Train on CIFAR-10"
	@echo "  clean       - Clean temporary files"
	@echo "  demo        - Run inference demo"
	@echo ""

# Install dependencies
install:
	pip install -r requirements.txt
	pip install -e .

# Run tests
test:
	python test_installation.py

# Run benchmarks
benchmark:
	python main.py benchmark

# Run training
train:
	python main.py train

# Run inference demo
demo:
	python main.py inference

# Clean temporary files
clean:
	rm -rf __pycache__
	rm -rf *.pyc
	rm -rf checkpoints/
	rm -rf data/
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Development setup
dev-setup:
	pip install -r requirements.txt
	pip install -e .
	pip install pytest black flake8 mypy

# Format code
format:
	black *.py
	flake8 *.py --max-line-length=100

# Type checking
type-check:
	mypy *.py --ignore-missing-imports