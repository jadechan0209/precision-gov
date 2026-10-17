
import os
import sys
import json
import hmac
import logging
import subprocess
from pathlib import Path

from flask import Flask, request, jsonify

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "scripts" / "batch_predict_v2.py"

API_KEY = os.environ.get("PREDICTION_API_KEY", "")

logging.basicConfig(level=logging.INFO)


@app.get("/")
def health_check():
    return jsonify({
        "service": "PRECISION-GOV Prediction API",
        "status": "online"
    }), 200


@app.post("/predict")
def predict():
    if not API_KEY:
        app.logger.error("Prediction API key is not configured.")
        return jsonify({
            "success": False,
            "error": "Service configuration error."
        }), 503

    supplied_key = request.headers.get("X-API-Key", "")

    if not hmac.compare_digest(supplied_key, API_KEY):
        return jsonify({
            "success": False,
            "error": "Unauthorized."
        }), 401

    records = request.get_json(silent=True)

    if not isinstance(records, list) or not records:
        return jsonify({
            "success": False,
            "error": "Expected a non-empty JSON array."
        }), 400

    if len(records) > 1000:
        return jsonify({
            "success": False,
            "error": "Maximum 1000 records per request."
        }), 400

    if not SCRIPT_PATH.is_file():
        app.logger.error("Prediction script is missing.")
        return jsonify({
            "success": False,
            "error": "Prediction service is unavailable."
        }), 500

    try:
        completed = subprocess.run(
            [sys.executable, str(SCRIPT_PATH)],
            input=json.dumps(records),
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
            timeout=180,
            check=False
        )

        if completed.returncode != 0:
            app.logger.error(
                "Prediction script failed: %s",
                completed.stderr[-4000:]
            )

            # The script may print a structured error to stdout.
            try:
                error_data = json.loads(completed.stdout)
                message = error_data.get(
                    "error", "Prediction failed."
                )
            except (ValueError, AttributeError):
                message = "Prediction failed."

            return jsonify({
                "success": False,
                "error": message
            }), 422

        result = json.loads(completed.stdout)

        if not isinstance(result, dict) or not result.get("success"):
            app.logger.error("Unexpected prediction response.")
            return jsonify({
                "success": False,
                "error": "Invalid prediction response."
            }), 500

        return jsonify(result), 200

    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "error": "Prediction timed out. Try a smaller batch."
        }), 504

    except (ValueError, OSError) as exc:
        app.logger.exception("Prediction service error: %s", exc)
        return jsonify({
            "success": False,
            "error": "Unable to complete prediction."
        }), 500
