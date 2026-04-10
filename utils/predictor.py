"""
utils/predictor.py
Loads the four XGBoost models and exposes a single predict() method
that accepts a raw request dict and returns a structured result dict.
"""

import os
import pickle
import warnings
import pandas as pd

from utils.encoders import encode_team, encode_venue, get_phase, PHASE_LABELS

warnings.filterwarnings("ignore")

# ── Model file paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


class CricketPredictor:
    """
    Loads all four pre-trained XGBoost models and provides
    ball-level and match-level predictions.

    Models
    ------
    dot_model      : DotBall.pkl          – P(dot ball)       binary
    boundary_model : BoundaryModel.pkl    – P(boundary 4/6)   binary
    run_model      : RunPrediction.pkl    – P(0..5 runs)      multi-class
    win_model      : IPLchasingTeamWin.pkl– P(chase success)  binary
    """

    def __init__(self):
        print("Loading models...")
        self.dot_model      = self._load("DotBall.pkl")
        self.boundary_model = self._load("BoundaryModel.pkl")
        self.run_model      = self._load("RunPrediction.pkl")
        self.win_model      = self._load("IPLchasingTeamWin.pkl")
        print("✓ All 4 models loaded successfully.\n")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(self, data: dict) -> dict:
        """
        Main prediction entry point.

        Parameters
        ----------
        data : dict   Raw JSON body from the API request.

        Returns
        -------
        dict with keys:
            dot_ball_prob    float  – % probability of a dot ball
            boundary_prob    float  – % probability of a 4 or 6
            expected_runs    float  – weighted average expected runs
            run_distribution list   – [P(0), P(1), P(2), P(3), P(4), P(5)]
            win_probability  float|None – % win prob for chasing team (innings=2 only)
            phase            str    – human-readable phase label
        """
        # ── 1. Parse & validate inputs ────────────────────────────────────────
        batting_team   = str(data.get("batting_team", ""))
        bowling_team   = str(data.get("bowling_team", ""))
        venue          = str(data.get("venue", ""))
        innings        = int(data.get("innings", 1))
        over           = int(data.get("over", 1))
        ball_in_over   = int(data.get("ball_in_over", 1))
        current_score  = float(data.get("current_score", 0))
        wickets_fallen = int(data.get("wickets_fallen", 0))

        # Optional performance features
        batter_sr      = float(data.get("batter_sr", 130))
        bowler_eco     = float(data.get("bowler_eco", 7.5))
        last_6_runs    = float(data.get("last_6_runs", 6))
        last_12_runs   = float(data.get("last_12_runs", 12))
        prev_runs      = float(data.get("prev_runs", 1))
        prev_wicket    = int(data.get("prev_wicket", 0))
        last_6_wickets = int(data.get("last_6_wickets", 0))
        striker_enc    = int(data.get("striker_enc", 0))
        bowler_enc     = int(data.get("bowler_enc", 0))

        # ── 2. Derived features ───────────────────────────────────────────────
        balls_bowled = over * 6 + ball_in_over
        run_rate     = round(current_score / max(balls_bowled, 1) * 6, 3)
        phase        = get_phase(over)

        # ── 3. Build ball-level feature DataFrame ─────────────────────────────
        ball_df = pd.DataFrame([{
            "striker_enc"      : striker_enc,
            "bowler_enc"       : bowler_enc,
            "batting_team_enc" : encode_team(batting_team),
            "bowling_team_enc" : encode_team(bowling_team),
            "venue_enc"        : encode_venue(venue),
            "over"             : over,
            "ball_in_over"     : ball_in_over,
            "phase"            : phase,
            "current_score"    : current_score,
            "wickets_fallen"   : wickets_fallen,
            "run_rate"         : run_rate,
            "prev_runs"        : prev_runs,
            "prev_wicket"      : prev_wicket,
            "last_6_runs"      : last_6_runs,
            "last_12_runs"     : last_12_runs,
            "last_6_wickets"   : last_6_wickets,
            "batter_sr"        : batter_sr,
            "bowler_eco"       : bowler_eco,
        }])

        # ── 4. Run the three ball-level models ────────────────────────────────
        dot_prob      = float(self.dot_model.predict_proba(ball_df)[0][1])
        boundary_prob = float(self.boundary_model.predict_proba(ball_df)[0][1])

        run_proba     = self.run_model.predict_proba(ball_df)[0]
        run_dist      = [round(float(p), 4) for p in run_proba]
        expected_runs = round(sum(i * run_dist[i] for i in range(len(run_dist))), 3)

        # ── 5. Win probability (2nd innings only) ─────────────────────────────
        win_prob = None
        if innings == 2:
            balls_remaining = int(data.get("balls_remaining", 60))
            balls_done_chase = max(120 - balls_remaining, 1)
            chase_rr = round(current_score / balls_done_chase * 6, 3)

            win_df = pd.DataFrame([{
                "batting_team"   : encode_team(batting_team),
                "bowling_team"   : encode_team(bowling_team),
                "venue"          : encode_venue(venue),
                "innings"        : innings,
                "current_score"  : current_score,
                "wickets_fallen" : wickets_fallen,
                "balls_remaining": balls_remaining,
                "run_rate"       : chase_rr,
            }])
            win_prob = round(float(self.win_model.predict_proba(win_df)[0][1]) * 100, 1)

        # ── 6. Return structured result ───────────────────────────────────────
        return {
            "dot_ball_prob"   : round(dot_prob * 100, 1),
            "boundary_prob"   : round(boundary_prob * 100, 1),
            "expected_runs"   : expected_runs,
            "run_distribution": run_dist,
            "win_probability" : win_prob,
            "phase"           : PHASE_LABELS[phase],
            "run_rate"        : round(run_rate, 2),
        }

    def models_loaded(self) -> list[str]:
        return ["DotBall", "BoundaryModel", "RunPrediction", "IPLchasingTeamWin"]

    # ── Private helpers ───────────────────────────────────────────────────────

    def _load(self, filename: str):
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}\n"
                f"Make sure {filename} is inside the 'models/' folder."
            )
        with open(path, "rb") as f:
            return pickle.load(f)
