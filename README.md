# Second Course in Statistical Learning

## Requirements

- Python 3.13
- [Poetry](https://python-poetry.org/docs/#installation)

## Setup

### macOS / Linux

```bash
make setup
```

### Windows

```cmd
make -f Makefile.win setup
```

This will:
1. Configure Poetry to create the virtual environment inside the project (`.venv/`)
2. Set Python 3.13 as the interpreter
3. Install all dependencies

## Activate the virtual environment

### macOS / Linux

```bash
make shell
```

### Windows

```cmd
make -f Makefile.win shell
```

## PyCharm Setup

After running `make setup`, configure PyCharm to use the project interpreter:

1. Open **Settings** → **Project** → **Python Interpreter**
2. Click **Add Interpreter** → **Add Local Interpreter**
3. Select **Existing** and point to `.venv/bin/python` (macOS/Linux) or `.venv\Scripts\python.exe` (Windows)

## Clean up

```bash
make clean          # macOS/Linux
make -f Makefile.win clean  # Windows
```
