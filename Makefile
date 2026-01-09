.PHONY: install install-dev run test lint format clean setup-repos

# Setup external repositories
setup-repos:
	bash scripts/setup_repos.sh


# Install production dependencies
install:
	pip install -r requirements.txt

# Install development dependencies
install-dev:
	pip install -r requirements-dev.txt

# Run the development server
run:
	uvicorn app.main:app --reload --port 8000

# Run tests with coverage
test:
	pytest tests/ -v --cov=app --cov-report=term-missing

# Run linter
lint:
	ruff check app/ tests/

# Format code
format:
	ruff format app/ tests/

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
