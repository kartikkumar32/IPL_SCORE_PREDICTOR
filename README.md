# 🏏 Cricket AI Predictor

A full-stack IPL ball-by-ball prediction system powered by 4 pre-trained XGBoost models.

## Features

| Page | Description |
|------|-------------|
| **Predict** | Input match data → get dot ball %, boundary %, run distribution, expected runs, and win probability |
| **Simulate** | Ball-by-ball T20 innings simulation with live ML predictions and win probability chart |
| **Model Info** | Technical details, feature tables, label encodings, and API reference |

---

## Quick Start in VS Code

### Step 1 — Open the project
```
File → Open Folder → select cricket_predictor/
```

### Step 2 — Create a virtual environment
Open the **VS Code Terminal** (`Ctrl+\``) and run:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Run the app

**Option A — VS Code Debugger (recommended):**
- Press `F5` or go to **Run → Start Debugging**
- Select **"Run Flask App"** configuration

**Option B — Terminal:**
```bash
python app.py
```

### Step 5 — Open in browser
```
http://127.0.0.1:5000
```

---

## Project Structure

```
cricket_predictor/
│
├── app.py                      # Flask application & routes
│
├── models/                     # Pre-trained XGBoost model files
│   ├── DotBall.pkl
│   ├── BoundaryModel.pkl
│   ├── RunPrediction.pkl
│   └── IPLchasingTeamWin.pkl
│
├── utils/
│   ├── __init__.py
│   ├── predictor.py            # CricketPredictor class (loads + runs all 4 models)
│   └── encoders.py             # Team/venue label encoding maps + phase logic
│
├── templates/
│   ├── base.html               # Shared navbar & layout
│   ├── index.html              # Prediction dashboard
│   ├── simulate.html           # Match simulation page
│   └── model_info.html         # Model details & API docs
│
├── static/
│   └── css/
│       └── style.css           # Full application stylesheet
│
├── .vscode/
│   ├── launch.json             # F5 debugger config
│   ├── settings.json           # Editor & Python settings
│   └── extensions.json         # Recommended extensions
│
├── requirements.txt
└── README.md
```

---

## Models

| File | Task | Input Features | Classes |
|------|------|---------------|---------|
| `DotBall.pkl` | Dot ball probability | 18 | Binary (0/1) |
| `BoundaryModel.pkl` | Boundary probability | 18 | Binary (0/1) |
| `RunPrediction.pkl` | Run distribution | 18 | Multi-class (0–5) |
| `IPLchasingTeamWin.pkl` | Win probability (2nd inn.) | 8 | Binary (0/1) |

### Ball model features (18)
`striker_enc` · `bowler_enc` · `batting_team_enc` · `bowling_team_enc` · `venue_enc` · `over` · `ball_in_over` · `phase` · `current_score` · `wickets_fallen` · `run_rate` · `prev_runs` · `prev_wicket` · `last_6_runs` · `last_12_runs` · `last_6_wickets` · `batter_sr` · `bowler_eco`

### Win model features (8)
`batting_team` · `bowling_team` · `venue` · `innings` · `current_score` · `wickets_fallen` · `balls_remaining` · `run_rate`

---

## API Endpoints

### `POST /api/predict`
```json
{
  "batting_team":   "Mumbai Indians",
  "bowling_team":   "Chennai Super Kings",
  "venue":          "Wankhede Stadium",
  "innings":        1,
  "over":           14,
  "ball_in_over":   3,
  "current_score":  110,
  "wickets_fallen": 2,
  "batter_sr":      148,
  "bowler_eco":     7.4,
  "last_6_runs":    11,
  "last_12_runs":   19
}
```

**Response:**
```json
{
  "dot_ball_prob":    24.3,
  "boundary_prob":    38.7,
  "expected_runs":    2.41,
  "run_distribution": [0.24, 0.22, 0.08, 0.06, 0.28, 0.12],
  "win_probability":  null,
  "phase":            "Middle Overs (Ov 7-15)",
  "run_rate":         7.86
}
```

### `GET /api/meta`
Returns available teams and venues.

### `GET /api/health`
Returns loaded model names and status.

---

## IPL Teams Supported
Chennai Super Kings · Delhi Capitals · Gujarat Titans · Kolkata Knight Riders ·
Lucknow Super Giants · Mumbai Indians · Punjab Kings · Rajasthan Royals ·
Royal Challengers Bangalore · Sunrisers Hyderabad

## Venues Supported
Arun Jaitley Stadium · Brabourne Stadium · DY Patil Stadium · Eden Gardens ·
Feroz Shah Kotla · MA Chidambaram Stadium · MCA Stadium ·
Maharashtra Cricket Association Stadium · Narendra Modi Stadium ·
Punjab Cricket Association Stadium · Rajiv Gandhi International Stadium ·
Sawai Mansingh Stadium · Wankhede Stadium
