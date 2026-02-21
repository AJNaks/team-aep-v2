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
) -> Dict[str, Any]:
    """
    Full categorisation pipeline.
    
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

    # Stage 1b + 2: LLM classification (with NIC lookup as context)
    llm_result = classify_with_llm(
        product_description or "",
        nic_desc,
        ondc_domains,
        nic_result,
    )

    llm_result["method"] = "nic_lookup+llm" if nic_result else "llm_only"
    llm_result["nic_code"] = nic_code
    llm_result["nic_description"] = nic_desc

    # Stage 3: If NIC lookup and LLM agree, boost confidence
    if nic_result and llm_result.get("primary_domain") == nic_result["ondc_domain"]:
        llm_result["confidence"] = min(
            1.0, max(llm_result.get("confidence", 0), nic_result["confidence"]) + 0.05
        )
        llm_result["reasoning"] += " (NIC lookup and LLM classification agree)"

    return llm_result
