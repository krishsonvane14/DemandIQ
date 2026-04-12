# DemandIQ

> **Intelligent demand forecasting** — load, clean, model, and evaluate demand data through a single, structured Python pipeline.

---

## Project Structure

```
DemandIQ/
├── app.py              # Pipeline entry point
├── config.py           # Shared settings, paths, and hyperparameters
├── requirements.txt    # Python dependencies
├── .gitignore
├── README.md
│
├── data/
│   ├── raw/            # Original, unmodified source data (git-ignored)
│   └── cleaned/        # Processed data ready for modelling (git-ignored)
│
├── notebooks/          # Exploratory notebooks (EDA, experiments)
│
└── src/
    ├── __init__.py
    ├── data_loader.py  # Loading & validating raw data
    ├── preprocessor.py # Cleaning & feature engineering
    ├── model.py        # Model definition, training, and serialisation
    └── evaluator.py    # Metrics, reports, and visualisations
```

---

## Quick Start

```bash
# 1. Clone and enter the repo
git clone <repo-url>
cd DemandIQ

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the pipeline
python app.py
```

---

## Configuration

Edit `config.py` to adjust data paths, model hyperparameters, and debug flags.  
Environment variables can be placed in a `.env` file (loaded automatically via `python-dotenv`).

---

## Development

- Place raw data files in `data/raw/` (they are git-ignored by default).
- Use `notebooks/` for exploratory analysis — checkpoints are also git-ignored.
- Implement each module in `src/` following the docstrings and `TODO` comments.

---

## License

See `LICENSE`.
