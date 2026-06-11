import json
import time
import requests
import google.auth
from google.auth.transport.requests import Request

# --- GCP VERTEX AI CONFIGURATION ---
project_id = 'ams-catalog-mngmt-dev-870'
location_id = 'europe-west1'
model = 'gemini-2.5-flash'

SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]
url = f"https://europe-west1-aiplatform.googleapis.com/v1beta1/projects/{project_id}/locations/{location_id}/publishers/google/models/{model}:generateContent"

# Initialize GCP Credentials
creds, _ = google.auth.default(scopes=SCOPES)
session = requests.Session()
session.verify = False

try:
    creds.refresh(Request(session=session))
except Exception as e:
    print(f"Warning on initial auth refresh: {e}")

# --- MIRAKL SCHEMA (Vertex AI REST Format) ---
mirakl_schema = {
    "type": "OBJECT",
    "properties": {
        "product_sku": {
            "type": "STRING",
            "description": "The unique identifier for the product. If not found, generate a logical one."
        },
        "category_code": {
            "type": "STRING",
            "description": "The Mirakl-specific taxonomy code (e.g., ELEC-001)"
        },
        "product_title": {
            "type": "STRING",
            "description": "Cleaned product title, max 100 characters"
        },
        "description": {
            "type": "STRING",
            "description": "Formatted product description"
        },
        "attributes": {
            "type": "OBJECT",
            "description": "Key-value pairs of specific Mirakl attribute codes and their values."
        }
    },
    "required": ["product_sku", "category_code", "product_title", "description", "attributes"]
}

# --- VERTEX AI CALL FUNCTION (From your script) ---


def call_gemini(prompt, max_retries=3, delay=2, response_schema=None):
    safety_settings = [
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"}
    ]

    generation_config = {
        "maxOutputTokens": 2048,
        "temperature": 0,
        "thinkingConfig": {"thinkingBudget": 0}
    }

    if response_schema:
        generation_config["responseMimeType"] = "application/json"
        generation_config["responseSchema"] = response_schema

    payload = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "safetySettings": safety_settings,
        "generationConfig": generation_config
    }

    # Ensure credentials are fresh before calling
    if not creds.valid:
        creds.refresh(Request(session=session))

    headers = {
        "Authorization": f"Bearer {creds.token}",
        "Content-Type": "application/json"
    }

    for attempt in range(1, max_retries + 1):
        try:
            r = session.post(url, headers=headers, json=payload, timeout=20)
            r.raise_for_status()
            data = r.json()

            if "candidates" in data and len(data["candidates"]) > 0:
                if "content" in data["candidates"][0] and "parts" in data["candidates"][0]["content"]:
                    return data["candidates"][0]["content"]["parts"][0]["text"]

            return '{"error": "Respuesta vacía o bloqueada por la API"}'

        except (requests.exceptions.RequestException, requests.exceptions.SSLError) as e:
            print(f"Intento {attempt} fallido: {e}")
            if attempt < max_retries:
                time.sleep(delay)
            else:
                return f'{{"error": "Error después de {max_retries} intentos: {e}"}}'

# --- MAPPING LOGIC ---


def retrieve_mirakl_rules() -> str:
    """Mock function for RAG context rules."""
    return """
    Mirakl Rules:
    - If the product is electronics, the category_code must be 'ELEC-001'.
    - If the product is furniture, the category_code must be 'FURN-002'.
    - Required attributes: 'COLOR', 'WEIGHT_KG', 'BRAND'.
    """


def map_text_to_mirakl(text_content: str) -> dict:
    taxonomy_rules = retrieve_mirakl_rules()

    prompt = f"""
    You are an expert e-commerce data mapper. Map the provided product sheet data into the exact Mirakl JSON schema. 
    Strictly follow these taxonomy rules: {taxonomy_rules}
    
    Product Data:
    {text_content}
    """

    # Call your Vertex AI function with the schema
    response_text = call_gemini(prompt=prompt, response_schema=mirakl_schema)

    # Parse the returned string into a Python dictionary
    try:
        mapped_data = json.loads(response_text)
        return mapped_data
    except json.JSONDecodeError:
        return {"error": "Failed to parse JSON", "raw_response": response_text}
