PYTHON = python3.13
POETRY = poetry
VENV_DIR = .venv

.PHONY: setup shell clean help

## setup: Create virtual environment, install all dependencies and Playwright browser
setup:
	$(POETRY) config virtualenvs.in-project true
	$(POETRY) env use $(PYTHON)
	$(POETRY) install
	$(POETRY) run playwright install chromium
	@echo ""
	@echo "Setup complete. Virtual environment created at $(VENV_DIR)/"
	@echo "Run 'make shell' to activate the virtual environment."

## shell: Activate the virtual environment
shell:
	@echo "Starting a shell with the virtual environment activated..."
	$(POETRY) shell

## clean: Remove the virtual environment and cached files
clean:
	rm -rf $(VENV_DIR)
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned virtual environment and cached files."

## help: Show this help message
help:
	@echo "Available targets:"
	@grep -E '^## ' Makefile | sed 's/## /  /'
