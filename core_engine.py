import os
import json
import base64
import wave
import io
import asyncio
import datetime
from PIL import Image
from google import genai
from google.genai import types
from toolbox_core import ToolboxClient
from utils import call_with_retry

USE_GROQ = os.environ.get("USE_GROQ", "false").lower() == "true"

client = genai.Client(
    vertexai=True,
    project=os.environ.get("GCP_PROJECT_ID", "project-52c4a541-a471-4a0f-807"),
    location="us-central1",
)
VISION_MODEL = "gemini-3.6-flash"
TEXT_MODEL = "gemini-3.6-flash"
TTS_MODEL = "gemini-2.5-flash-preview-tts"  # kept on Gemini regardless of USE_GROQ
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSIONS = 256  # small + fast, plenty for similarity over a personal history

TOOLBOX_URL = os.environ.get("TOOLBOX_URL", "http://127.0.0.1:5000")

if USE_GROQ:
    from groq import Groq
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    GROQ_VISION_MODEL = "qwen/qwen3.6-27b"  # multimodal (reasoning + vision), Groq preview tier
    GROQ_TEXT_MODEL = "openai/gpt-oss-120b"  # production tier


# 0. Embeddings + retrieval (RAG over this business's own past analyses)
def get_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list:
    """Turns text into a vector. task_type='RETRIEVAL_DOCUMENT' for things
    being stored, 'RETRIEVAL_QUERY' for the thing you're searching with."""
    if DEMO_MODE:
        # Deterministic fake vector based on text hash, so similarity
        # comparisons still behave sanely (same text -> same vector)
        # without calling the embeddings API.
        import random
        rng = random.Random(text)
        return [rng.uniform(-0.5, 0.5) for _ in range(EMBEDDING_DIMENSIONS)]

    response = call_with_retry(
        client.models.embed_content,
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSIONS,
        ),
    )
    return list(response.embeddings[0].values)


def cosine_similarity(a: list, b: list) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(y * y for y in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def retrieve_similar_past_situation(current_summary: str, history: list, similarity_threshold: float = 0.75):
    """Given a text summary of the current batch and a list of past history
    entries (each with 'embedding' and 'advice'), finds the most similar
    past situation, if any is similar enough to be worth referencing."""
    entries_with_embeddings = [h for h in history if h.get("embedding")]
    if not entries_with_embeddings:
        return None

    query_vec = get_embedding(current_summary, task_type="RETRIEVAL_QUERY")
    scored = [
        (cosine_similarity(query_vec, h["embedding"]), h)
        for h in entries_with_embeddings
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best = scored[0]

    if best_score < similarity_threshold:
        return None

    return {
        "advice": best["advice"],
        "timestamp": best.get("timestamp"),
        "outflow_percentage": best.get("outflow_percentage"),
        "similarity": round(best_score, 3),
    }


DEMO_MODE = os.environ.get("DEMO_MODE", "false").lower() == "true"


# 1. Vision extraction -- ONE receipt per call, so a busy or mixed-currency
# batch can't cause the model to drop a receipt or blend currencies together.
def extract_single_receipt(uploaded_file) -> str:
    if DEMO_MODE:
        # Canned, realistic-looking receipt data so the pipeline runs
        # end-to-end without hitting the Gemini API. Rotates through a
        # couple of plausible tailor-shop receipts based on filename hash
        # so a multi-file demo upload doesn't look identical every time.
        import random
        samples = [
            '{"currency_detected": "INR", "original_total": 4500, "total_outflow_inr": 4500, "transactions": [{"description": "Silk fabric bulk order", "amount_inr": 4500}]}',
            '{"currency_detected": "INR", "original_total": 1800, "total_outflow_inr": 1800, "transactions": [{"description": "Thread and trims", "amount_inr": 1800}]}',
            '{"currency_detected": "USD", "original_total": 60, "total_outflow_inr": 4980, "transactions": [{"description": "Imported sewing machine parts", "amount_inr": 4980}], "conversion_note": "60 USD -> INR 4980"}',
        ]
        random.seed(getattr(uploaded_file, "filename", "demo"))
        return random.choice(samples)

    prompt = """
    You are an expert AI financial extractor analyzing ONE receipt or invoice image.
    Return ONLY a valid JSON object, no markdown formatting. It must contain:
    - "currency_detected": the currency shown on the receipt, e.g. "INR", "USD", "EUR"
    - "original_total": the total amount exactly as printed, in its original currency
    - "total_outflow_inr": the total converted to INR. If currency_detected is already
      INR, this equals original_total unchanged. If it is USD, convert using
      1 USD = 83 INR. If it is EUR, convert using 1 EUR = 90 INR. For any other
      currency, use your best reasonable estimate and say so in a "conversion_note" field.
    - "transactions": a list of objects, each with "description" (string) and
      "amount_inr" (number, already converted to INR using the same rate).
    Only extract what's actually on THIS receipt. Do not estimate or invent totals.
    """

    if USE_GROQ:
        uploaded_file.seek(0)
        img_bytes = uploaded_file.read()
        uploaded_file.seek(0)
        b64_img = base64.b64encode(img_bytes).decode("utf-8")
        content_type = getattr(uploaded_file, "content_type", None) or "image/jpeg"
        response = call_with_retry(
            groq_client.chat.completions.create,
            model=GROQ_VISION_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{b64_img}"}},
                ],
            }],
            temperature=0.1,
            max_tokens=600,
            reasoning_effort="none",
        )
        raw_text = response.choices[0].message.content
        return raw_text.replace("```json", "").replace("```", "").strip()

    image = Image.open(uploaded_file)
    contents = [image, prompt]
    response = call_with_retry(
        client.models.generate_content,
        model=VISION_MODEL,
        contents=contents
    )
    clean_json = response.text.replace("```json", "").replace("```", "").strip()
    return clean_json


def extract_json_from_batch(uploaded_files):
    """Returns a LIST of JSON strings, one per receipt (not one combined blob).
    main.py should pass this list straight into generate_financial_advice
    without wrapping it in another list."""
    return [extract_single_receipt(f) for f in uploaded_files]


# 2. The Multi-Invoice Aggregator -- does the actual addition in Python,
# not inside a single model call, so nothing gets silently dropped.
def aggregate_invoices(invoice_jsons):
    total_outflow = 0
    total_inflow = 0
    all_transactions = []
    receipts_seen = 0
    conversion_notes = []

    for invoice in invoice_jsons:
        data = json.loads(invoice)
        receipts_seen += 1

        outflow = data.get('total_outflow_inr', data.get('total_outflow', 0)) or 0
        total_outflow += outflow
        total_inflow += data.get('total_inflow_inr', data.get('total_inflow', 0)) or 0

        if 'transactions' in data:
            all_transactions.extend(data['transactions'])

        if data.get('currency_detected') and data['currency_detected'] != 'INR':
            note = f"{data.get('original_total')} {data['currency_detected']} -> INR {outflow}"
            conversion_notes.append(data.get('conversion_note', note))

    return {
        "total_outflow": total_outflow,
        "total_inflow": total_inflow,
        "transactions_count": len(all_transactions),
        "receipts_processed": receipts_seen,
        "currency_conversions": conversion_notes,
    }


# 3. Business Profile Lookup via MCP Toolbox
async def _fetch_profile_via_toolbox(business_id: str) -> dict:
    async with ToolboxClient(TOOLBOX_URL) as toolbox_client:
        tool = await toolbox_client.load_tool("get-business-profile")
        result = await tool(business_id=business_id)
        print(f"[MCP Toolbox DEBUG] raw result type={type(result)!r} value={result!r}")

        parsed = result if isinstance(result, (list, dict)) else json.loads(result)
        if not parsed:
            raise ValueError(f"No profile found for business_id={business_id!r} (empty result from toolbox)")
        # The toolbox returns a single JSON object when the query has LIMIT 1,
        # not a list-of-one -- handle both shapes.
        row = parsed[0] if isinstance(parsed, list) else parsed
        return {
            "cash_flow_category": row["cash_flow_category"],
            "monthly_turnover_inr": row["monthly_turnover_inr"],
            "upi_transaction_percentage": row["upi_transaction_percentage"],
            "default_risk": row["default_risk"],
        }


def get_business_profile(business_id: str) -> dict:
    print(f"[MCP Toolbox]: Requesting profile for {business_id}")
    try:
        return asyncio.run(_fetch_profile_via_toolbox(business_id))
    except Exception as e:
        print(f"[MCP Toolbox Warning]: Falling back due to: {str(e)}")
        return {
            "cash_flow_category": "Steady",
            "monthly_turnover_inr": 85000,
            "upi_transaction_percentage": 75,
            "default_risk": "Low",
        }


# 4. Trust Tier / Badge logic
def compute_risk_tier(outflow_percentage: float, default_risk: str) -> str:
    risk = (default_risk or "").lower()
    if risk == "low" and outflow_percentage < 60:
        return "Gold"
    elif risk in ("low", "medium") and outflow_percentage < 100:
        return "Silver"
    else:
        return "Bronze"


# 4b. Forecast / loan-readiness -- single source of truth, do not also
# define this in main.py (that caused a real import bug earlier).
def compute_forecast(history):
    """Simple linear trend over past outflow percentages, used to project
    the next likely outflow and a rough loan-readiness score. Needs at
    least 2 past analyses to say anything meaningful."""
    if len(history) < 2:
        return None

    n = len(history)
    xs = list(range(n))
    ys = [h["outflow_percentage"] for h in history]
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    denom = sum((x - x_mean) ** 2 for x in xs) or 1
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean

    forecast_next = round(slope * n + intercept, 1)
    loan_readiness = round(max(0, min(100, 100 - forecast_next)), 1)

    return {
        "forecast_next_outflow_pct": forecast_next,
        "trend": "rising" if slope > 1 else "falling" if slope < -1 else "steady",
        "loan_readiness_score": loan_readiness,
    }


# 5. The Generalized Coaching Function
def generate_financial_advice(dynamic_business_id, user_language, invoice_list, user_profession="Tailor", retrieved_context=None):
    aggregated_data = aggregate_invoices(invoice_list)
    profile = get_business_profile(dynamic_business_id)
    current_month = datetime.datetime.now().strftime("%B")

    turnover = profile["monthly_turnover_inr"] or 1
    outflow_percentage = round((aggregated_data["total_outflow"] / turnover) * 100, 1)
    risk_tier = compute_risk_tier(outflow_percentage, profile["default_risk"])

    if DEMO_MODE:
        advice_text = (
            f"Your recent spending comes to {outflow_percentage}% of your typical monthly turnover of "
            f"₹{turnover:,}, across {aggregated_data['receipts_processed']} receipt(s) this batch. "
            f"That keeps you in a healthy range for a {user_profession.lower()} business.\n\n"
            f"With {current_month} approaching, this is a strong time to stock up on wedding-season "
            f"and festival-ready materials -- demand for tailoring work typically rises sharply in the "
            f"weeks before major festivals, so inventory bought now positions you to take on more orders "
            f"without a last-minute scramble.\n\n"
            f"Keep tracking your outflow against turnover each week. If it stays under 60%, you're "
            f"building a strong, low-risk pattern that improves your loan readiness over time."
        )
        return {
            "advice": advice_text,
            "outflow_percentage": outflow_percentage,
            "risk_tier": risk_tier,
            "turnover": turnover,
            "profile": profile,
            "aggregated_data": aggregated_data,
            "retrieved_context": retrieved_context,
        }

    retrieved_block = ""
    if retrieved_context:
        retrieved_block = f"""
    RETRIEVED PAST SITUATION (from this same business's history, {retrieved_context['similarity']*100:.0f}% similar to now):
    On a previous occasion with {retrieved_context['outflow_percentage']}% outflow, you advised: "{retrieved_context['advice'][:400]}"
    If genuinely relevant, briefly reference what's changed or stayed the same since then. Don't force it if it doesn't add value.
    """

    prompt = f"""
    You are Sahay, a financial coach for Indian micro-entrepreneurs.
    The user's Business ID is {dynamic_business_id}.
    Their preferred language is {user_language}.
    Their profession is: {user_profession}
    The current month is: {current_month}

    Business Profile: {profile}
    Aggregated Invoice Data (all amounts already in INR): {aggregated_data}
    This batch's outflow is {outflow_percentage}% of their typical monthly turnover.
    {aggregated_data['receipts_processed']} receipt(s) were processed in this batch.
    {retrieved_block}

    Your task is to provide financial advice based on the data above. All figures
    given to you are already in INR -- do not attempt currency conversion yourself,
    it has already been done.

    STEP 1: Analyze the expense against their cash flow. State the percentage.
    STEP 2: Identify any upcoming Indian cultural events, festivals, or seasonal shifts relevant to the current month.
    STEP 3: Provide strategic business advice tailored SPECIFICALLY to a {user_profession}. Explain how this invoice helps them capture the upcoming seasonal demand.

    Keep the tone professional, direct, and supportive. Do not use markdown formatting like bolding or bullet points unless necessary for structure.
    """

    if USE_GROQ:
        response = call_with_retry(
            groq_client.chat.completions.create,
            model=GROQ_TEXT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=600,
            reasoning_effort="low",
        )
        advice_text = response.choices[0].message.content
    else:
        response = call_with_retry(
            client.models.generate_content,
            model=TEXT_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        advice_text = response.text

    return {
        "advice": advice_text,
        "outflow_percentage": outflow_percentage,
        "risk_tier": risk_tier,
        "turnover": turnover,
        "profile": profile,
        "aggregated_data": aggregated_data,
        "retrieved_context": retrieved_context,
    }


# 6. Text-to-Speech coaching audio
def generate_advice_audio(text: str, voice_name: str = "Kore") -> bytes:
    if DEMO_MODE:
        # Silent placeholder audio so the pipeline completes without
        # calling the TTS API. Swap for a real gTTS/offline call if you
        # want actual spoken audio in the demo.
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            wf.writeframes(b"\x00\x00" * 24000)  # 1 second of silence
        return buffer.getvalue()

    response = call_with_retry(
        client.models.generate_content,
        model=TTS_MODEL,
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        ),
    )
    pcm_data = response.candidates[0].content.parts[0].inline_data.data
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)
        wf.writeframes(pcm_data)
    return buffer.getvalue()


if __name__ == "__main__":
    result = generate_financial_advice(
        dynamic_business_id="BIZ_1042",
        user_language="Hindi",
        invoice_list=[
            '{"currency_detected": "USD", "original_total": 322.0, "total_outflow_inr": 26726, "transactions": [{"description": "Wedding gown alterations", "amount_inr": 26726}]}',
            '{"currency_detected": "INR", "original_total": 160291.2, "total_outflow_inr": 160291.2, "transactions": [{"description": "Trims and thread bulk order", "amount_inr": 160291.2}]}'
        ]
    )
    print(result["advice"])
    print(f"\nOutflow %: {result['outflow_percentage']} | Tier: {result['risk_tier']}")
    print(result["aggregated_data"])