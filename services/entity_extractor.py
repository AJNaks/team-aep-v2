"""
Entity Extraction Pipeline
---------------------------
Three-layer extraction: Regex → LLM → API Validation
Extracts business entities from voice transcripts (potentially code-switched Hindi-English)
"""

import os
import re
import json
import anthropic
from typing import Dict, Optional, Any


# Indian identifier patterns
PATTERNS = {
    "udyam": r"UDYAM[-\s]?[A-Z]{2}[-\s]?\d{2}[-\s]?\d{7}",
    "gst": r"\d{2}[A-Z]{5}\d{4}[A-Z]{1}\d{1}[A-Z]{1}[A-Z\d]{1}",
    "pan": r"[A-Z]{5}\d{4}[A-Z]{1}",
    "pincode": r"\b[1-9]\d{5}\b",
    "mobile": r"\b[6-9]\d{9}\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
}


def extract_with_regex(text: str) -> Dict[str, Optional[str]]:
    """Layer 1: Fast deterministic extraction using regex patterns."""
    # Normalize text - remove extra spaces, uppercase for ID matching
    normalized = text.upper().replace(" - ", "-").replace("  ", " ")
    
    results = {}
    for field, pattern in PATTERNS.items():
        match = re.search(pattern, normalized)
        results[field] = match.group(0) if match else None
    
    return results


def extract_with_llm(transcript: str, regex_results: Dict) -> Dict[str, Any]:
    """Layer 2: LLM-based semantic extraction for fields regex missed."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    # Build context about what regex already found
    found = {k: v for k, v in regex_results.items() if v}
    missing = [k for k, v in regex_results.items() if not v]
    
    prompt = f"""You are an entity extraction system for Indian MSE (Micro & Small Enterprise) registration.

Extract business information from this voice transcript. The speaker may use Hindi, English, or a mix.

TRANSCRIPT: "{transcript}"

ALREADY EXTRACTED (verified by regex): {json.dumps(found)}

STILL NEEDED: {json.dumps(missing)}

Additionally, extract these fields if mentioned:
- enterprise_name: Business/company/shop name
- owner_name: Owner's full name
- product_description: What they manufacture/sell (in English)
- state: Indian state
- district: District name
- business_activity: Brief description of business activity

Return ONLY valid JSON with all fields. Use null for fields not found in the transcript.
Do NOT invent or guess values. Only extract what is explicitly stated.

JSON:"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}]
    )
    
    try:
        # Parse LLM response - handle potential markdown wrapping
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {}


def validate_with_udyam(udyam_number: str, mock_data: Dict) -> Optional[Dict]:
    """Layer 3: Validate and enrich using Udyam API (mock for PoC)."""
    if not udyam_number:
        return None
    
    # Normalize the number
    normalized = udyam_number.upper().replace(" ", "").replace("–", "-")
    
    # Look up in mock data
    enterprise = mock_data.get("enterprises", {}).get(normalized)
    return enterprise


def extract_entities(transcript: str, mock_udyam_data: Dict) -> Dict[str, Any]:
    """
    Full extraction pipeline: Regex → LLM → API Validation
    Returns merged, validated entity data.
    """
    # Layer 1: Regex
    regex_results = extract_with_regex(transcript)
    
    # Layer 2: LLM extraction
    llm_results = extract_with_llm(transcript, regex_results)
    
    # Merge: regex takes priority for identifier fields (more reliable)
    merged = {**llm_results}
    for field, value in regex_results.items():
        if value:
            merged[field] = value
    
    # Layer 3: Udyam API validation and auto-fill
    udyam_number = merged.get("udyam")
    udyam_data = validate_with_udyam(udyam_number, mock_udyam_data)
    
    if udyam_data:
        merged["udyam_verified"] = True
        merged["auto_filled"] = {}
        # Auto-fill from Udyam - only fill if not already present
        auto_fill_map = {
            "enterprise_name": "enterprise_name",
            "owner_name": "owner_name",
            "enterprise_type": "enterprise_type",
            "nic_5digit": "nic_5digit",
            "nic_description": "nic_description",
            "state": "state",
            "district": "district",
            "pincode": "pincode",
            "gst_number": "gst_number",
            "mobile": "mobile",
            "email": "email",
        }
        for target_field, source_field in auto_fill_map.items():
            if not merged.get(target_field) and udyam_data.get(source_field):
                merged[target_field] = udyam_data[source_field]
                merged["auto_filled"][target_field] = "udyam_api"
    else:
        merged["udyam_verified"] = False
        merged["auto_filled"] = {}
    
    return merged
