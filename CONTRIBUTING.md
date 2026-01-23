# Contributing to Credit Scoring Engine

Thank you for your interest in contributing to the Credit Scoring Engine!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/AdiSk325/credit-scoring-engine.git
cd credit-scoring-engine
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

## Running Tests

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=credit_scoring --cov-report=html
```

## Code Style

We follow PEP 8 style guidelines. Before submitting a PR:

```bash
# Format code
black credit_scoring/

# Check style
flake8 credit_scoring/
```

## Pull Request Process

1. Create a new branch for your feature
2. Write tests for your changes
3. Ensure all tests pass
4. Update documentation as needed
5. Submit a pull request

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on the code, not the person
