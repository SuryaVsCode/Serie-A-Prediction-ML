

from flask import Flask, request, jsonify, render_template
import pickle, json
import numpy as np
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

ART_MODEL = MODEL_DIR / "FootballPrediction.pkl"
ART_COLS  = MODEL_DIR / "feature_columns.json"
ART_PROF  = MODEL_DIR / "team_profiles.json"


# --- Load artifacts ---
if not (ART_MODEL.exists() and ART_COLS.exists() and ART_PROF.exists()):
    raise RuntimeError("Missing artifacts. Ensure you ran the training export cell.")

with open(ART_MODEL, "rb") as f:
    model = pickle.load(f)

with open(ART_COLS, "r") as f:
    feature_columns = json.load(f)["feature_columns"]

with open(ART_PROF, "r") as f:
    team_profiles = json.load(f)["profiles"]

# Derive home/away feature buckets from the saved columns
home_cols = [c for c in feature_columns if c.endswith("_home")]
away_cols = [c for c in feature_columns if c.endswith("_away")]

# Flask app
app = Flask(__name__)

def build_feature_row(home_team_id: int, away_team_id: int) -> pd.DataFrame:
    """Build a single-row DataFrame in the exact feature order."""
    hkey, akey = str(home_team_id), str(away_team_id)

    if hkey not in team_profiles:
        raise ValueError(f"Unknown home teamId: {home_team_id}")
    if akey not in team_profiles:
        raise ValueError(f"Unknown away teamId: {away_team_id}")

    home_vec = dict(team_profiles[hkey]["home"])
    away_vec = dict(team_profiles[akey]["away"])

    # Guarantee id columns are set (safety)
    if "teamId_home" in home_cols:
        home_vec["teamId_home"] = float(home_team_id)
    if "teamId_away" in away_cols:
        away_vec["teamId_away"] = float(away_team_id)

    # Merge to one dict, then to DF
    row = {**home_vec, **away_vec}

    # Reindex to exact training order, fill any accidental gaps with 0.0
    df_row = pd.DataFrame([row]).reindex(columns=feature_columns)
    df_row = df_row.fillna(0.0)

    # Ensure numeric dtype
    for c in df_row.columns:
        df_row[c] = pd.to_numeric(df_row[c], errors="coerce").fillna(0.0)

    return df_row

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

teams_df = pd.read_csv("training/teams.csv")
team_map = dict(zip(teams_df["teamId"], teams_df["name"]))

@app.route("/api/teams", methods=["GET"])
def api_teams():
    ids = sorted(int(tid) for tid in team_profiles.keys())
    teams = [{"id": tid, "name": team_map.get(tid, f"Team {tid}")} for tid in ids]
    return jsonify({"teams": teams})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    try:
        home_id = int(payload.get("homeTeamId"))
        away_id = int(payload.get("awayTeamId"))
    except Exception:
        return jsonify({"error": "Invalid team ids"}), 400

    if home_id == away_id:
        return jsonify({"error": "Home and Away teams must be different."}), 400

    try:
        features = build_feature_row(home_id, away_id)
        pred = model.predict(features)[0]
        # Try probabilities if supported
        probs = None
        if hasattr(model, "predict_proba"):
            # class order inferred from model.classes_
            proba = model.predict_proba(features)[0]
            class_order = list(getattr(model, "classes_", [0,1,2]))
            probs = {int(cls): float(p) for cls, p in zip(class_order, proba)}
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    label_map = {0: "Away Win", 1: "Home Win", 2: "Draw"}
    result = label_map.get(int(pred), str(pred))

    return jsonify({
        "prediction": result,
        "raw_class": int(pred),
        "probabilities": probs
    })

if __name__ == "__main__":
    # In production, use gunicorn/uwsgi; for dev:
    app.run(host="0.0.0.0", port=5000, debug=True)
