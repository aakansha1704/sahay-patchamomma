import os
import uuid
import datetime
from flask import Flask, render_template, request, jsonify, session, send_file
from google.cloud import storage, firestore
from werkzeug.security import generate_password_hash, check_password_hash
from core_engine import (
    generate_financial_advice, generate_advice_audio, extract_json_from_batch,
    aggregate_invoices, get_embedding, retrieve_similar_past_situation,compute_forecast,
)

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")

GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "sahay-invoices-archive")
GCP_PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "project-52c4a541-a471-4a0f-807")

db = firestore.Client(project=GCP_PROJECT_ID, database="dbdb")
USERS_COLLECTION = "users"
ANALYSES_COLLECTION = "analyses"
MESSAGES_COLLECTION = "messages"

# Legacy demo accounts, kept as a safety net so your existing demo login
# (BIZ_1042 / sahay123) still works even if Firestore is unreachable.
DEMO_USERS = {
    "BIZ_1042": {"password": "sahay123", "name": "Priya", "profession": "Tailor"},
    "BIZ_2091": {"password": "sahay456", "name": "Arjun", "profession": "Kirana Store Owner"},
}


def get_gcs_client():
    return storage.Client()


def upload_to_gcs(file_storage, business_id: str) -> str:
    client = get_gcs_client()
    bucket = client.bucket(GCS_BUCKET_NAME)
    blob_name = f"{business_id}/{uuid.uuid4().hex}_{file_storage.filename}"
    blob = bucket.blob(blob_name)
    file_storage.seek(0)
    blob.upload_from_file(file_storage, content_type=file_storage.content_type)
    file_storage.seek(0)
    return f"gs://{GCS_BUCKET_NAME}/{blob_name}"


def _new_business_id() -> str:
    return f"BIZ_{uuid.uuid4().hex[:6].upper()}"


@app.route("/")
def index():
    return render_template("index.html")


# --- Email/password signup & login (self-contained, no external setup) ----
@app.route("/api/signup", methods=["POST"])
def signup():
    data = request.get_json(force=True)
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    name = (data.get("name") or "").strip()
    profession = data.get("profession") or "Tailor"

    if not email or not password or not name:
        return jsonify({"error": "Name, email, and password are all required."}), 400

    user_ref = db.collection(USERS_COLLECTION).document(email)
    if user_ref.get().exists:
        return jsonify({"error": "An account with that email already exists. Log in instead."}), 409

    business_id = _new_business_id()
    user_ref.set({
        "email": email,
        "password_hash": generate_password_hash(password),
        "name": name,
        "profession": profession,
        "business_id": business_id,
    })

    session["email"] = email
    session["business_id"] = business_id
    return jsonify({"email": email, "business_id": business_id, "name": name, "profession": profession})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json(force=True)
    identifier = (data.get("business_id") or "").strip()
    password = data.get("password") or ""

    # Try email/Firestore account first
    email_key = identifier.lower()
    user_ref = db.collection(USERS_COLLECTION).document(email_key)
    user_doc = user_ref.get()
    if user_doc.exists:
        user = user_doc.to_dict()
        if check_password_hash(user["password_hash"], password):
            session["email"] = email_key
            session["business_id"] = user["business_id"]
            return jsonify({
                "business_id": user["business_id"],
                "name": user["name"],
                "profession": user["profession"],
            })
        return jsonify({"error": "Incorrect password."}), 401

    # Fall back to legacy demo business-ID accounts
    demo_user = DEMO_USERS.get(identifier)
    if demo_user and demo_user["password"] == password:
        session["business_id"] = identifier
        session.pop("email", None)
        return jsonify({
            "business_id": identifier,
            "name": demo_user["name"],
            "profession": demo_user["profession"],
        })

    return jsonify({"error": "No account found with that email/business ID and password."}), 401


# --- Google sign-in ---------------------------------------------------------
# Requires a Firebase project: console.firebase.google.com -> create project
# -> Authentication -> enable Google sign-in provider -> copy the web config
# into templates/index.html where marked. This route verifies the ID token
# the frontend gets back from that Firebase popup.
@app.route("/api/google-login", methods=["POST"])
def google_login():
    try:
        import firebase_admin
        from firebase_admin import auth as firebase_auth
        if not firebase_admin._apps:
            firebase_admin.initialize_app(options={"projectId": "project-52c4a541-a471-4a0f-807"})
    except Exception as e:
        return jsonify({"error": f"Google sign-in isn't configured yet: {e}"}), 501

    id_token = request.get_json(force=True).get("id_token")
    if not id_token:
        return jsonify({"error": "Missing ID token."}), 400

    try:
        decoded = firebase_auth.verify_id_token(id_token)
    except Exception as e:
        return jsonify({"error": f"Invalid Google sign-in token: {e}"}), 401

    email = decoded.get("email", "").lower()
    name = decoded.get("name", email.split("@")[0])

    user_ref = db.collection(USERS_COLLECTION).document(email)
    user_doc = user_ref.get()
    if user_doc.exists:
        user = user_doc.to_dict()
    else:
        business_id = _new_business_id()
        user = {"email": email, "name": name, "profession": "Tailor", "business_id": business_id}
        user_ref.set(user)

    session["email"] = email
    session["business_id"] = user["business_id"]
    return jsonify({"business_id": user["business_id"], "name": user["name"], "profession": user["profession"]})


@app.route("/api/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/analyze", methods=["POST"])
def analyze():
    business_id = session.get("business_id")
    if not business_id:
        return jsonify({"error": "Not logged in."}), 401

    files = request.files.getlist("files")
    language = request.form.get("language", "English")
    profession = request.form.get("profession", "Tailor")

    if not files:
        return jsonify({"error": "Upload at least one invoice image."}), 400

    try:
        gcs_paths = []
        for f in files:
            try:
                gcs_paths.append(upload_to_gcs(f, business_id))
            except Exception as gcs_err:
                print(f"[GCS Warning]: Could not archive {f.filename}: {gcs_err}")

        # extract_json_from_batch now returns a LIST of per-receipt JSON strings
        live_batch_data = extract_json_from_batch(files)
        aggregated_preview = aggregate_invoices(live_batch_data)

        # --- RAG: retrieve a similar past situation for this business, if any ---
        history_ref = db.collection(ANALYSES_COLLECTION).document(business_id).collection("entries")
        past_entries = [d.to_dict() for d in history_ref.stream()]
        retrieved_context = None
        try:
            summary_text = (
                f"{profession} business, outflow of {aggregated_preview['total_outflow']} INR "
                f"across {aggregated_preview['receipts_processed']} receipts."
            )
            retrieved_context = retrieve_similar_past_situation(summary_text, past_entries)
        except Exception as rag_err:
            print(f"[RAG Warning]: Retrieval skipped due to: {rag_err}")

        result = generate_financial_advice(business_id, language, live_batch_data, profession, retrieved_context)

        audio_url = None
        try:
            audio_bytes = generate_advice_audio(result["advice"])
            audio_path = f"/tmp/{uuid.uuid4().hex}.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            session["last_audio_path"] = audio_path
            audio_url = "/api/audio"
        except Exception as audio_err:
            print(f"[TTS Warning]: {audio_err}")

        # Save this analysis to history (with its embedding, for future RAG
        # retrieval), then compute a forecast from all past analyses.
        try:
            entry_embedding = get_embedding(
                f"{profession} business, outflow of {result['aggregated_data']['total_outflow']} INR, "
                f"{result['outflow_percentage']}% of turnover. {result['advice'][:500]}"
            )
        except Exception as embed_err:
            print(f"[RAG Warning]: Could not embed this entry: {embed_err}")
            entry_embedding = None

        history_ref.add({
            "outflow_percentage": result["outflow_percentage"],
            "turnover": result["turnover"],
            "risk_tier": result["risk_tier"],
            "advice": result["advice"],
            "embedding": entry_embedding,
            "timestamp": datetime.datetime.utcnow().isoformat(),
        })
        history_docs = history_ref.order_by("timestamp").stream()
        history = [d.to_dict() for d in history_docs]
        forecast = compute_forecast(history)

        return jsonify({
            "advice": result["advice"],
            "risk_tier": result["risk_tier"],
            "turnover": result["turnover"],
            "outflow_percentage": result["outflow_percentage"],
            "audio_url": audio_url,
            "archived_receipts": gcs_paths,
            "receipts_processed": result["aggregated_data"]["receipts_processed"],
            "currency_conversions": result["aggregated_data"]["currency_conversions"],
            "forecast": forecast,
            "history": history,
            "retrieved_context": result.get("retrieved_context"),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/history")
def get_history():
    business_id = session.get("business_id")
    if not business_id:
        return jsonify({"error": "Not logged in."}), 401
    history_ref = db.collection(ANALYSES_COLLECTION).document(business_id).collection("entries")
    docs = history_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(10).stream()
    return jsonify({"history": [d.to_dict() for d in docs]})


@app.route("/api/profile", methods=["GET", "POST"])
def profile():
    business_id = session.get("business_id")
    email = session.get("email")
    if not business_id:
        return jsonify({"error": "Not logged in."}), 401

    if request.method == "GET":
        if email:
            user_doc = db.collection(USERS_COLLECTION).document(email).get()
            if user_doc.exists:
                return jsonify(user_doc.to_dict())
        demo = DEMO_USERS.get(business_id)
        if demo:
            return jsonify({"business_id": business_id, **demo})
        return jsonify({"business_id": business_id, "name": "", "profession": "Tailor"})

    # POST -- update name/profession
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    profession = data.get("profession") or "Tailor"

    if email:
        db.collection(USERS_COLLECTION).document(email).set(
            {"name": name, "profession": profession}, merge=True
        )
    return jsonify({"business_id": business_id, "name": name, "profession": profession})


@app.route("/api/conversations")
def conversations():
    business_id = session.get("business_id")
    if not business_id:
        return jsonify({"error": "Not logged in."}), 401

    history_ref = db.collection(ANALYSES_COLLECTION).document(business_id).collection("entries")
    docs = history_ref.order_by("timestamp", direction=firestore.Query.DESCENDING).limit(20).stream()
    return jsonify([d.to_dict() for d in docs])


@app.route("/api/contact", methods=["POST"])
def contact():
    business_id = session.get("business_id", "anonymous")
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()
    contact_email = (data.get("email") or session.get("email") or "").strip()

    if not message:
        return jsonify({"error": "Message can't be empty."}), 400

    db.collection(MESSAGES_COLLECTION).add({
        "business_id": business_id,
        "email": contact_email,
        "message": message,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    })
    return jsonify({"ok": True})


@app.route("/api/audio")
def get_audio():
    audio_path = session.get("last_audio_path")
    if not audio_path or not os.path.exists(audio_path):
        return jsonify({"error": "No audio available."}), 404
    return send_file(audio_path, mimetype="audio/wav")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)