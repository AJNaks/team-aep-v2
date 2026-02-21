"""
MSE-SNP Matching Engine
------------------------
6-factor weighted scoring:
1. Domain alignment (0.30)
2. Geographic coverage (0.20)
3. Digital readiness fit (0.15)
4. SNP success rate (0.15)
5. Capacity availability (0.10)
6. Language support (0.10)
"""

import json
from typing import Dict, List, Any, Optional


# Scoring weights
WEIGHTS = {
    "domain": 0.30,
    "geography": 0.20,
    "readiness": 0.15,
    "success_rate": 0.15,
    "capacity": 0.10,
    "language": 0.10,
}


def score_domain(mse_domain: str, snp: Dict) -> float:
    """Score 0-1 based on domain alignment."""
    if mse_domain == snp.get("primary_domain"):
        return 1.0
    if mse_domain in snp.get("ondc_domains", []):
        return 0.7
    return 0.0


def score_geography(mse_state: str, snp: Dict) -> float:
    """Score 0-1 based on geographic coverage."""
    if mse_state in snp.get("geography", []):
        return 1.0
    return 0.0


def score_readiness(mse_readiness: int, snp: Dict) -> float:
    """Score 0-1 based on whether MSE meets SNP's minimum readiness threshold."""
    min_score = snp.get("min_readiness_score", 0)
    if mse_readiness >= min_score:
        # Higher score if MSE exceeds minimum by a good margin
        excess = mse_readiness - min_score
        return min(1.0, 0.7 + (excess / 100) * 0.3)
    else:
        # Penalty proportional to gap
        gap = min_score - mse_readiness
        return max(0.0, 1.0 - (gap / 50))


def score_success_rate(snp: Dict) -> float:
    """Score 0-1 based on SNP's historical success rate."""
    return snp.get("success_rate", 0.5)


def score_capacity(snp: Dict) -> float:
    """Score 0-1 based on remaining capacity."""
    current = snp.get("current_sellers", 0)
    max_cap = snp.get("max_capacity", 1)
    utilisation = current / max_cap if max_cap > 0 else 1.0
    if utilisation >= 0.95:
        return 0.1  # Nearly full
    elif utilisation >= 0.80:
        return 0.5
    else:
        return 1.0


def score_language(mse_language: str, snp: Dict) -> float:
    """Score 0-1 based on language support."""
    if mse_language in snp.get("languages", []):
        return 1.0
    if "en" in snp.get("languages", []) or "hi" in snp.get("languages", []):
        return 0.4  # Fallback to Hindi/English
    return 0.0


def estimate_digital_readiness(mse_data: Dict) -> int:
    """
    Estimate MSE's digital readiness score (0-100) from available signals.
    This is a key differentiator - inferring readiness from behaviour.
    """
    score = 0
    
    # Has GST registration (+20)
    if mse_data.get("gst_number"):
        score += 20
    
    # Has email (+10)
    if mse_data.get("email"):
        score += 10
    
    # Has Udyam registration (+15)
    if mse_data.get("udyam"):
        score += 15
    
    # Enterprise type signals
    etype = mse_data.get("enterprise_type", "").lower()
    if etype == "small":
        score += 15  # Small enterprises tend to be more digitally mature
    elif etype == "micro":
        score += 5
    
    # Turnover as proxy for operational sophistication
    turnover = mse_data.get("turnover", 0)
    if turnover > 10000000:  # > 1 crore
        score += 20
    elif turnover > 5000000:  # > 50 lakhs
        score += 15
    elif turnover > 1000000:  # > 10 lakhs
        score += 10
    else:
        score += 5
    
    # Input channel used (would be tracked in production)
    channel = mse_data.get("input_channel", "voice")
    if channel == "web":
        score += 15
    elif channel == "whatsapp":
        score += 10
    elif channel == "voice":
        score += 5
    
    return min(100, score)


def match_mse_to_snps(
    mse_data: Dict,
    snp_profiles_path: str,
    top_n: int = 3,
) -> List[Dict[str, Any]]:
    """
    Match an MSE to the best SNPs using 6-factor weighted scoring.
    
    Args:
        mse_data: Must contain: ondc_domain, state, language (ISO code)
                  Optional: digital_readiness, enterprise_type, turnover, gst_number
        snp_profiles_path: Path to SNP profiles JSON
        top_n: Number of top recommendations to return
    
    Returns:
        List of top-N SNP matches with scores and explanations
    """
    with open(snp_profiles_path, "r") as f:
        snp_data = json.load(f)
    
    mse_domain = mse_data.get("ondc_domain", "")
    mse_state = mse_data.get("state", "")
    mse_language = mse_data.get("language", "hi")
    
    # Estimate digital readiness if not provided
    mse_readiness = mse_data.get("digital_readiness")
    if mse_readiness is None:
        mse_readiness = estimate_digital_readiness(mse_data)
    
    results = []
    
    for snp in snp_data.get("snps", []):
        # Calculate individual scores
        scores = {
            "domain": score_domain(mse_domain, snp),
            "geography": score_geography(mse_state, snp),
            "readiness": score_readiness(mse_readiness, snp),
            "success_rate": score_success_rate(snp),
            "capacity": score_capacity(snp),
            "language": score_language(mse_language, snp),
        }
        
        # Weighted total
        total = sum(scores[k] * WEIGHTS[k] for k in WEIGHTS)
        
        # Generate explainable reasons
        reasons = []
        if scores["domain"] >= 0.7:
            reasons.append(f"Supports {mse_domain} ({snp.get('speciality', '')})")
        if scores["geography"] == 1.0:
            reasons.append(f"Operates in {mse_state}")
        if scores["success_rate"] >= 0.75:
            reasons.append(f"{int(snp['success_rate']*100)}% seller success rate")
        if scores["language"] == 1.0:
            reasons.append(f"Supports {mse_language} language")
        if snp.get("cataloguing_support"):
            reasons.append("Provides cataloguing support")
        
        # Flags
        flags = []
        if scores["domain"] == 0:
            flags.append("Domain mismatch")
        if scores["geography"] == 0:
            flags.append("Not in MSE's state")
        if scores["capacity"] <= 0.1:
            flags.append("Near capacity")
        
        results.append({
            "snp_id": snp["snp_id"],
            "snp_name": snp["name"],
            "total_score": round(total, 3),
            "factor_scores": {k: round(v, 2) for k, v in scores.items()},
            "reasons": reasons,
            "flags": flags,
            "avg_onboarding_days": snp.get("avg_onboarding_days"),
            "commission_rate": snp.get("commission_rate"),
            "support_level": snp.get("support_level"),
            "cataloguing_support": snp.get("cataloguing_support"),
        })
    
    # Sort by total score descending
    results.sort(key=lambda x: x["total_score"], reverse=True)
    
    # Add rank and recommendation strength
    for i, r in enumerate(results[:top_n]):
        r["rank"] = i + 1
        if r["total_score"] >= 0.7:
            r["recommendation"] = "Strong Match"
        elif r["total_score"] >= 0.5:
            r["recommendation"] = "Good Match"
        elif r["total_score"] >= 0.3:
            r["recommendation"] = "Possible Match"
        else:
            r["recommendation"] = "Weak Match"
    
    return results[:top_n]
