"""
Product Categorisation Engine
------------------------------
Three-stage pipeline:
1. Signal Fusion: NIC code lookup + text classification
2. Taxonomy Mapping: NIC → ONDC domain with confidence scoring
3. Multi-domain Resolution: Handle ambiguous mappings
"""

import os
import json
import anthropic
from typing import Dict, List, Optional, Any


def load_mapping(mapping_path: str) -> Dict:
    """Load NIC-to-ONDC mapping data."""
    with open(mapping_path, "r") as f:
        return json.load(f)


def lookup_nic_code(nic_code: str, mapping_data: Dict) -> Optional[Dict]:
    """Stage 1a: Direct NIC code lookup in curated mapping table."""
    for entry in mapping_data.get("nic_to_ondc", []):
        if entry["nic_code"] == nic_code:
            return entry
    return None


def classify_with_llm(
    product_description: str,
    nic_description: str,
    ondc_domains: Dict,
    nic_lookup_result: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Stage 1b + 2: LLM-based classification when NIC lookup is ambiguous
    or needs confirmation. Also handles multi-domain resolution.
    """
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    domains_desc = "\n".join(
        f"- {code}: {info['name']} ({info['description']})"
        for code, info in ondc_domains.items()
    )

    context = ""
    if nic_lookup_result:
        context = (
            f"\nNIC lookup suggests: {nic_lookup_result['ondc_domain']} "
            f"({nic_lookup_result['ondc_category']}) with confidence {nic_lookup_result['confidence']}"
            f"\nNotes: {nic_lookup_result.get('notes', 'N/A')}"
        )

    prompt = f"""You are an ONDC product categorisation expert. Map this MSE's products to ONDC retail domains.

NIC DESCRIPTION: {nic_description}
PRODUCT DESCRIPTION: {product_description}
{context}

AVAILABLE ONDC DOMAINS:
{domains_desc}

Respond with ONLY valid JSON:
{{
    "primary_domain": "RETXX",
    "primary_category": "category name",
    "primary_subcategory": "subcategory name",
    "confidence": 0.XX,
    "secondary_domain": "RETXX or null",
    "secondary_confidence": 0.XX,
    "reasoning": "brief explanation of why this mapping was chosen",
    "ambiguous": true/false,
    "suggested_attributes": ["attr1", "attr2"]
}}

Be precise. If the product clearly maps to one domain, set ambiguous=false and confidence high.
If it could map to multiple domains, set ambiguous=true and provide secondary_domain.
JSON:"""

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )

    try:
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return {
            "primary_domain": "UNKNOWN",
            "confidence": 0.0,
            "reasoning": "Failed to parse LLM response",
            "ambiguous": True,
        }


def categorise_mse(
    nic_code: str,
    product_description: str,
    mapping_path: str,
    image_analysis: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Full multi-signal categorisation pipeline.

    Signals:
        1. NIC code → direct lookup in curated mapping
        2. Product description → LLM classification
        3. Image analysis (optional) → VLM-detected category + extracted fields

    Returns:
        Dict with primary_domain, confidence, reasoning, and optional secondary mappings
    """
    mapping_data = load_mapping(mapping_path)
    ondc_domains = mapping_data.get("ondc_domains", {})

    # Stage 1a: NIC code direct lookup
    nic_result = lookup_nic_code(nic_code, mapping_data)

    nic_desc = nic_result["nic_desc"] if nic_result else f"NIC {nic_code}"

    # If high confidence direct match and no product description to refine
    if nic_result and nic_result["confidence"] >= 0.90 and not product_description:
        return {
            "primary_domain": nic_result["ondc_domain"],
            "primary_category": nic_result["ondc_category"],
            "primary_subcategory": nic_result["ondc_subcategory"],
            "confidence": nic_result["confidence"],
            "secondary_domain": None,
            "secondary_confidence": 0.0,
            "reasoning": f"High-confidence NIC lookup: {nic_result['notes']}",
            "ambiguous": False,
            "method": "nic_lookup",
            "nic_code": nic_code,
            "nic_description": nic_desc,
        }

    # Stage 1c: Image analysis signal (if available)
    image_desc = ""
    image_category = None
    if image_analysis:
        image_category = image_analysis.get("detected_category")
        image_fields = image_analysis.get("extracted_fields", {})
        image_desc_parts = []
        if image_analysis.get("description"):
            image_desc_parts.append(f"Image description: {image_analysis['description']}")
        if image_fields:
            field_str = ", ".join(f"{k}: {v}" for k, v in image_fields.items() if v)
            image_desc_parts.append(f"Extracted from photo: {field_str}")
        image_desc = ". ".join(image_desc_parts)

    # Combine all text signals for LLM
    combined_desc = " ".join(filter(None, [product_description, image_desc]))

    # Stage 1b + 2: LLM classification (with NIC lookup + image as context)
    llm_result = classify_with_llm(
        combined_desc or "",
        nic_desc,
        ondc_domains,
        nic_result,
    )

    # Build method string showing which signals were used
    signals = []
    if nic_result:
        signals.append("nic_lookup")
    if product_description:
        signals.append("text")
    if image_analysis:
        signals.append("image")
    signals.append("llm")
    llm_result["method"] = "+".join(signals)
    llm_result["nic_code"] = nic_code
    llm_result["nic_description"] = nic_desc

    # Stage 3: Multi-signal agreement boosting
    agreement_count = 0

    # Check NIC ↔ LLM agreement
    if nic_result and llm_result.get("primary_domain") == nic_result["ondc_domain"]:
        agreement_count += 1
        llm_result["confidence"] = min(
            1.0, max(llm_result.get("confidence", 0), nic_result["confidence"]) + 0.05
        )
        llm_result["reasoning"] += " (NIC lookup and LLM agree)"

    # Check image ↔ LLM agreement
    if image_category:
        # Map image category to expected domain
        cat_domain_map = {
            "food": "ONDC:RET10", "food_packaged": "ONDC:RET10",
            "textiles": "ONDC:RET12", "textiles_handloom": "ONDC:RET12",
            "handicrafts": "ONDC:RET14", "agriculture": "ONDC:RET10",
        }
        expected_domain = cat_domain_map.get(image_category)
        if expected_domain and expected_domain == llm_result.get("primary_domain"):
            agreement_count += 1
            llm_result["confidence"] = min(1.0, llm_result.get("confidence", 0) + 0.05)
            llm_result["reasoning"] += " (Image analysis confirms category)"

    # Triple agreement → highest confidence
    if agreement_count >= 2:
        llm_result["confidence"] = min(1.0, llm_result.get("confidence", 0) + 0.05)
        llm_result["ambiguous"] = False

    return llm_result
