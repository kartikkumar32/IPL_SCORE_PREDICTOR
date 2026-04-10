"""
Cricket AI Predictor - Main Application
Flask web application for IPL ball-by-ball predictions
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from utils.predictor import CricketPredictor
from utils.encoders import TEAMS, VENUES
import logging

# ─────────────────────────────────────────────
#  APP SETUP
# ─────────────────────────────────────────────

app = Flask(__name__)
CORS(app)  # Allow frontend/API access

logging.basicConfig(level=logging.INFO)

predictor = CricketPredictor()


# ─────────────────────────────────────────────
#  PAGES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", teams=TEAMS, venues=VENUES)


@app.route("/simulate")
def simulate():
    return render_template("simulate.html", teams=TEAMS, venues=VENUES)


@app.route("/model-info")
def model_info():
    return render_template("model_info.html")


# ─────────────────────────────────────────────
#  API ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Ball-level prediction endpoint"""
    try:
        data = request.get_json(force=True)

        if not data:
            return jsonify({"error": "No JSON body received"}), 400

        # ✅ Required fields check
        required_fields = [
            "batting_team", "bowling_team", "venue",
            "innings", "over", "ball_in_over",
            "current_score", "wickets_fallen"
        ]

        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({
                "error": f"Missing fields: {', '.join(missing)}"
            }), 400

        # ✅ Default values (safe fallback)
        data.setdefault("batter_sr", 130)
        data.setdefault("bowler_eco", 7.5)
        data.setdefault("last_6_runs", 6)
        data.setdefault("last_12_runs", 12)
        data.setdefault("prev_runs", 1)
        data.setdefault("prev_wicket", 0)
        data.setdefault("last_6_wickets", 0)
        data.setdefault("striker_enc", 0)
        data.setdefault("bowler_enc", 0)

        # ✅ Special check for 2nd innings
        if int(data["innings"]) == 2 and "balls_remaining" not in data:
            return jsonify({
                "error": "balls_remaining required for innings=2"
            }), 400

        result = predictor.predict(data)

        return jsonify({
            "success": True,
            "prediction": result
        })

    except ValueError as e:
        logging.error(f"ValueError: {str(e)}")
        return jsonify({"error": str(e)}), 422

    except Exception as e:
        logging.exception("Prediction failed")
        return jsonify({
            "error": "Internal server error",
            "details": str(e)
        }), 500


@app.route("/api/meta", methods=["GET"])
def api_meta():
    """Returns available teams and venues"""
    return jsonify({
        "teams": TEAMS,
        "venues": VENUES
    })


@app.route("/api/health", methods=["GET"])
def api_health():
    """Health check"""
    return jsonify({
        "status": "ok",
        "models_loaded": predictor.models_loaded()
    })


# ─────────────────────────────────────────────
#  RUN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("🏏 Cricket AI Predictor")
    print("Running at → http://127.0.0.1:5000")
    print("=" * 55 + "\n")

    app.run(debug=True, port=5000)