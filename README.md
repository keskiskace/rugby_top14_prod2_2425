# Top 14 Rugby Players Explorer

An interactive [Streamlit](https://streamlit.io/) application for exploring player statistics from the 2024–2025 French Top 14 rugby season. The app lets you inspect individual performance, build custom XVs, compare with a reference team and search for similar players.

## Features
- **Player dashboard** – browse detailed statistics for every player in the database.
- **Custom team builder** – assemble your own XV and visualise aggregate metrics on radar charts.
- **Team comparison** – compare your selection with a preset reference team.
- **Similarity search** – find players with comparable profiles using weighted statistics.

Data is scraped from public sources and provided for educational purposes only.

## Requirements
- Python 3.10 or later
- Packages listed in [`requirements.txt`](requirements.txt)
- Local copy of the `top14_players.db` SQLite database

## Installation
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd rugby_top14_2425
   ```
2. **Create a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Ensure the database path is correct**
   The app looks for a SQLite file specified by the `DB_FILE` constant in [`app.py`](app.py).  By default it points to `top14_players.db` located at the project root. Update the path if you store the database elsewhere.

## Usage
Start the Streamlit application:
```bash
streamlit run app.py
```
The app opens in your default browser. Use the sidebar controls to pick players, adjust statistics and explore the Top 14 landscape.

## Repository structure
```
rugby_top14_2425/
├── app.py               # Streamlit application
├── images/              # Player photos used by the app
├── top14_players.db     # SQLite database of player statistics
└── requirements.txt     # Python dependencies
```

## Acknowledgements
This project is an unofficial educational tool. All player data is obtained from publicly available sources.  The authors are not affiliated with the French Rugby Federation or the Top 14 organisers.
