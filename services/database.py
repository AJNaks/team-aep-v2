"""
TEAM-AEP v2 — SQLite Database Layer
====================================
Handles all persistence: MSME sessions, SNP profiles, MSE-SNP assignments,
claims, NSIC officials, OTP sessions.
"""

import sqlite3
import json
import secrets
from pathlib import Path
from datetime import datetime, timedelta

DB_PATH = Path(__file__).parent.parent / "data" / "team_aep.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS msme_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    udyam_number TEXT NOT NULL UNIQUE,
    current_step TEXT DEFAULT 'verify',
    language TEXT DEFAULT 'en',
    enterprise_data TEXT,
    categorisation_data TEXT,
    selected_snp TEXT,
    catalogue_data TEXT,
    completed_at TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS snp_profiles (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snp_id TEXT UNIQUE NOT NULL,
    name TEXT,
    email TEXT,
    subscriber_id TEXT,
    mobile TEXT,
    profile_data TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS mse_snp_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    udyam_number TEXT NOT NULL,
    snp_id TEXT NOT NULL,
    status TEXT DEFAULT 'assigned',
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(udyam_number, snp_id)
);

CREATE TABLE IF NOT EXISTS claims (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snp_id TEXT NOT NULL,
    udyam_number TEXT NOT NULL,
    claim_type TEXT NOT NULL,
    amount REAL,
    sku_count INTEGER,
    evidence TEXT,
    status TEXT DEFAULT 'pending',
    reviewed_by TEXT,
    review_notes TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reviewed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS nsic_officials (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT UNIQUE NOT NULL,
    name TEXT,
    mobile TEXT,
    role TEXT DEFAULT 'verifier',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS otp_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    identifier TEXT NOT NULL,
    role TEXT NOT NULL,
    otp_code TEXT NOT NULL,
    verified INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS document_verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    udyam_number TEXT NOT NULL,
    document_type TEXT NOT NULL,
    extracted_data TEXT,
    verification_result TEXT,
    raw_ocr_text TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    conn = get_db()
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    conn.close()


# ── OTP Sessions ─────────────────────────────────────────

def create_otp_session(identifier: str, role: str) -> str:
    otp_code = f"{secrets.randbelow(900000) + 100000}"
    expires_at = (datetime.utcnow() + timedelta(minutes=10)).isoformat()
    conn = get_db()
    # Remove any existing OTP for this identifier+role
    conn.execute("DELETE FROM otp_sessions WHERE identifier=? AND role=?", (identifier, role))
    conn.execute(
        "INSERT INTO otp_sessions (identifier, role, otp_code, expires_at) VALUES (?,?,?,?)",
        (identifier, role, otp_code, expires_at),
    )
    conn.commit()
    conn.close()
    return otp_code


def verify_otp_session(identifier: str, role: str, otp_code: str) -> bool:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM otp_sessions WHERE identifier=? AND role=? AND otp_code=? AND verified=0",
        (identifier, role, otp_code),
    ).fetchone()
    if not row:
        conn.close()
        return False
    # Check expiry
    if row["expires_at"] and datetime.fromisoformat(row["expires_at"]) < datetime.utcnow():
        conn.close()
        return False
    conn.execute("UPDATE otp_sessions SET verified=1 WHERE id=?", (row["id"],))
    conn.commit()
    conn.close()
    return True


# ── SNP Profiles ─────────────────────────────────────────

def get_snp_by_email(email: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM snp_profiles WHERE email=?", (email,)).fetchone()
    conn.close()
    if not row:
        return None
    return _snp_row_to_dict(row)


def get_snp_by_subscriber_id(subscriber_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM snp_profiles WHERE subscriber_id=?", (subscriber_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return _snp_row_to_dict(row)


def get_snp_by_id(snp_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM snp_profiles WHERE snp_id=?", (snp_id,)).fetchone()
    conn.close()
    if not row:
        return None
    return _snp_row_to_dict(row)


def get_all_snps() -> list:
    conn = get_db()
    rows = conn.execute("SELECT * FROM snp_profiles ORDER BY snp_id").fetchall()
    conn.close()
    return [_snp_row_to_dict(r) for r in rows]


def _snp_row_to_dict(row) -> dict:
    d = dict(row)
    if d.get("profile_data"):
        d["profile_data"] = json.loads(d["profile_data"])
    return d


# ── NSIC Officials ───────────────────────────────────────

def get_nsic_by_employee_id(employee_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM nsic_officials WHERE employee_id=?", (employee_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


# ── MSME Sessions ────────────────────────────────────────

def get_msme_session(udyam_number: str) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM msme_sessions WHERE udyam_number=?", (udyam_number,)).fetchone()
    conn.close()
    if not row:
        return None
    d = dict(row)
    for field in ("enterprise_data", "categorisation_data", "selected_snp", "catalogue_data"):
        if d.get(field):
            d[field] = json.loads(d[field])
    return d


def save_msme_progress(udyam_number: str, step: str, data: dict):
    conn = get_db()
    existing = conn.execute("SELECT id FROM msme_sessions WHERE udyam_number=?", (udyam_number,)).fetchone()

    now = datetime.utcnow().isoformat()
    enterprise_data = json.dumps(data.get("enterprise_data")) if data.get("enterprise_data") else None
    categorisation_data = json.dumps(data.get("categorisation_data")) if data.get("categorisation_data") else None
    selected_snp = json.dumps(data.get("selected_snp")) if data.get("selected_snp") else None
    catalogue_data = json.dumps(data.get("catalogue_data")) if data.get("catalogue_data") else None
    language = data.get("language", "en")
    completed_at = now if step == "summary" else None

    if existing:
        conn.execute("""
            UPDATE msme_sessions SET
                current_step=?, language=?,
                enterprise_data=COALESCE(?, enterprise_data),
                categorisation_data=COALESCE(?, categorisation_data),
                selected_snp=COALESCE(?, selected_snp),
                catalogue_data=COALESCE(?, catalogue_data),
                completed_at=COALESCE(?, completed_at),
                updated_at=?
            WHERE udyam_number=?
        """, (step, language, enterprise_data, categorisation_data, selected_snp, catalogue_data, completed_at, now, udyam_number))
    else:
        conn.execute("""
            INSERT INTO msme_sessions (udyam_number, current_step, language, enterprise_data, categorisation_data, selected_snp, catalogue_data, completed_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (udyam_number, step, language, enterprise_data, categorisation_data, selected_snp, catalogue_data, completed_at, now))

    conn.commit()
    conn.close()


# ── MSE-SNP Assignments ─────────────────────────────────

def get_assignments_for_snp(snp_id: str) -> list:
    conn = get_db()
    rows = conn.execute("""
        SELECT a.*, s.enterprise_data, s.current_step, s.catalogue_data, s.categorisation_data
        FROM mse_snp_assignments a
        LEFT JOIN msme_sessions s ON a.udyam_number = s.udyam_number
        WHERE a.snp_id=?
        ORDER BY a.assigned_at DESC
    """, (snp_id,)).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        for field in ("enterprise_data", "catalogue_data", "categorisation_data"):
            if d.get(field):
                d[field] = json.loads(d[field])
        result.append(d)
    return result


def get_assignment(udyam_number: str, snp_id: str) -> dict | None:
    conn = get_db()
    row = conn.execute(
        "SELECT * FROM mse_snp_assignments WHERE udyam_number=? AND snp_id=?",
        (udyam_number, snp_id),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def assign_mse_to_snp(udyam_number: str, snp_id: str):
    conn = get_db()
    conn.execute(
        "INSERT OR IGNORE INTO mse_snp_assignments (udyam_number, snp_id) VALUES (?,?)",
        (udyam_number, snp_id),
    )
    conn.commit()
    conn.close()


def update_assignment_status(udyam_number: str, snp_id: str, status: str):
    conn = get_db()
    conn.execute(
        "UPDATE mse_snp_assignments SET status=? WHERE udyam_number=? AND snp_id=?",
        (status, udyam_number, snp_id),
    )
    conn.commit()
    conn.close()


# ── Claims ───────────────────────────────────────────────

def get_claims_for_snp(snp_id: str) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM claims WHERE snp_id=? ORDER BY submitted_at DESC", (snp_id,)
    ).fetchall()
    conn.close()
    return [_claim_row_to_dict(r) for r in rows]


def get_all_claims(status: str | None = None) -> list:
    conn = get_db()
    if status:
        rows = conn.execute("SELECT * FROM claims WHERE status=? ORDER BY submitted_at DESC", (status,)).fetchall()
    else:
        rows = conn.execute("SELECT * FROM claims ORDER BY submitted_at DESC").fetchall()
    conn.close()
    return [_claim_row_to_dict(r) for r in rows]


def get_claim_by_id(claim_id: int) -> dict | None:
    conn = get_db()
    row = conn.execute("SELECT * FROM claims WHERE id=?", (claim_id,)).fetchone()
    conn.close()
    return _claim_row_to_dict(row) if row else None


def submit_claim(snp_id: str, udyam_number: str, claim_type: str, amount: float, sku_count: int = 0, evidence: dict = None):
    conn = get_db()
    conn.execute(
        "INSERT INTO claims (snp_id, udyam_number, claim_type, amount, sku_count, evidence) VALUES (?,?,?,?,?,?)",
        (snp_id, udyam_number, claim_type, amount, sku_count, json.dumps(evidence or {})),
    )
    conn.commit()
    conn.close()


def review_claim(claim_id: int, employee_id: str, status: str, notes: str = ""):
    conn = get_db()
    now = datetime.utcnow().isoformat()
    conn.execute(
        "UPDATE claims SET status=?, reviewed_by=?, review_notes=?, reviewed_at=? WHERE id=?",
        (status, employee_id, notes, now, claim_id),
    )
    conn.commit()
    conn.close()


def _claim_row_to_dict(row) -> dict:
    d = dict(row)
    if d.get("evidence"):
        d["evidence"] = json.loads(d["evidence"])
    return d


# ── Analytics / Dashboards ───────────────────────────────

def get_snp_dashboard_stats(snp_id: str) -> dict:
    conn = get_db()
    total = conn.execute("SELECT COUNT(*) FROM mse_snp_assignments WHERE snp_id=?", (snp_id,)).fetchone()[0]
    catalogue_pending = conn.execute(
        "SELECT COUNT(*) FROM mse_snp_assignments WHERE snp_id=? AND status IN ('assigned','onboarding','catalogue_pending')",
        (snp_id,),
    ).fetchone()[0]
    active = conn.execute(
        "SELECT COUNT(*) FROM mse_snp_assignments WHERE snp_id=? AND status='live'", (snp_id,)
    ).fetchone()[0]
    total_claims = conn.execute("SELECT COUNT(*) FROM claims WHERE snp_id=?", (snp_id,)).fetchone()[0]
    approved_claims = conn.execute(
        "SELECT COUNT(*) FROM claims WHERE snp_id=? AND status='approved'", (snp_id,)
    ).fetchone()[0]
    total_amount = conn.execute(
        "SELECT COALESCE(SUM(amount),0) FROM claims WHERE snp_id=?", (snp_id,)
    ).fetchone()[0]
    approved_amount = conn.execute(
        "SELECT COALESCE(SUM(amount),0) FROM claims WHERE snp_id=? AND status='approved'", (snp_id,)
    ).fetchone()[0]
    conn.close()
    return {
        "total_assigned": total,
        "catalogue_pending": catalogue_pending,
        "active_mses": active,
        "total_claims": total_claims,
        "approved_claims": approved_claims,
        "total_claimed_amount": total_amount,
        "total_approved_amount": approved_amount,
    }


def get_platform_analytics() -> dict:
    conn = get_db()
    total_mses = conn.execute("SELECT COUNT(*) FROM msme_sessions").fetchone()[0]
    total_snps = conn.execute("SELECT COUNT(*) FROM snp_profiles").fetchone()[0]
    claims_pending = conn.execute("SELECT COUNT(*) FROM claims WHERE status='pending'").fetchone()[0]
    claims_approved = conn.execute("SELECT COUNT(*) FROM claims WHERE status='approved'").fetchone()[0]
    claims_rejected = conn.execute("SELECT COUNT(*) FROM claims WHERE status='rejected'").fetchone()[0]
    total_disbursed = conn.execute(
        "SELECT COALESCE(SUM(amount),0) FROM claims WHERE status='approved'"
    ).fetchone()[0]

    # Geographic breakdown from enterprise_data
    mse_rows = conn.execute("SELECT enterprise_data FROM msme_sessions WHERE enterprise_data IS NOT NULL").fetchall()
    geo = {}
    women_count = 0
    for r in mse_rows:
        ent = json.loads(r["enterprise_data"]) if r["enterprise_data"] else {}
        state = ent.get("state", "Unknown")
        geo[state] = geo.get(state, 0) + 1
        if ent.get("gender", "").lower() == "female":
            women_count += 1

    conn.close()
    return {
        "total_mses": total_mses,
        "total_snps": total_snps,
        "claims_pending": claims_pending,
        "claims_approved": claims_approved,
        "claims_rejected": claims_rejected,
        "total_disbursed": total_disbursed,
        "women_owned_pct": round(women_count / max(total_mses, 1), 2),
        "geographic_breakdown": geo,
    }


def get_all_mses(status: str = None, state: str = None) -> list:
    conn = get_db()
    query = """
        SELECT s.*, a.snp_id, a.status as assignment_status
        FROM msme_sessions s
        LEFT JOIN mse_snp_assignments a ON s.udyam_number = a.udyam_number
        WHERE 1=1
    """
    params = []
    if status:
        query += " AND a.status=?"
        params.append(status)
    query += " ORDER BY s.updated_at DESC"

    rows = conn.execute(query, params).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        for field in ("enterprise_data", "categorisation_data", "selected_snp", "catalogue_data"):
            if d.get(field):
                d[field] = json.loads(d[field])
        # Filter by state if requested (enterprise_data.state)
        if state and d.get("enterprise_data", {}).get("state") != state:
            continue
        result.append(d)
    return result


def get_snps_with_metrics() -> list:
    conn = get_db()
    snps = conn.execute("SELECT * FROM snp_profiles ORDER BY snp_id").fetchall()
    result = []
    for s in snps:
        d = _snp_row_to_dict(s)
        snp_id = d["snp_id"]
        d["mses_assigned"] = conn.execute(
            "SELECT COUNT(*) FROM mse_snp_assignments WHERE snp_id=?", (snp_id,)
        ).fetchone()[0]
        d["mses_live"] = conn.execute(
            "SELECT COUNT(*) FROM mse_snp_assignments WHERE snp_id=? AND status='live'", (snp_id,)
        ).fetchone()[0]
        d["claims_submitted"] = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE snp_id=?", (snp_id,)
        ).fetchone()[0]
        d["claims_approved"] = conn.execute(
            "SELECT COUNT(*) FROM claims WHERE snp_id=? AND status='approved'", (snp_id,)
        ).fetchone()[0]
        d["total_claimed"] = conn.execute(
            "SELECT COALESCE(SUM(amount),0) FROM claims WHERE snp_id=?", (snp_id,)
        ).fetchone()[0]
        result.append(d)
    conn.close()
    return result


def check_claim_eligibility(udyam_number: str) -> dict:
    """Check TEAM eligibility for an MSE based on stored enterprise data."""
    session = get_msme_session(udyam_number)
    ent = session.get("enterprise_data", {}) if session else {}

    etype = ent.get("enterprise_type", "")
    is_micro_or_small = etype.lower() in ("micro", "small")

    # NIC 2-digit codes for manufacturing: 10-33, services: others
    nic_2 = ent.get("nic_2digit", "")
    major_activity = "Manufacturing" if nic_2 and 10 <= int(nic_2) <= 33 else "Services"
    is_mfg_or_services = major_activity in ("Manufacturing", "Services")

    not_on_ondc = True  # Mock: assume not already on ONDC

    # Check if at least 2 delivered orders (mock: check claims evidence)
    conn = get_db()
    acct_claims = conn.execute(
        "SELECT evidence FROM claims WHERE udyam_number=? AND claim_type='account_management'",
        (udyam_number,),
    ).fetchall()
    conn.close()
    delivered_orders = 0
    for c in acct_claims:
        ev = json.loads(c["evidence"]) if c["evidence"] else {}
        delivered_orders += ev.get("delivered_orders", 0)
    min_orders_met = delivered_orders >= 2

    return {
        "enterprise_type": etype,
        "is_micro_or_small": is_micro_or_small,
        "major_activity": major_activity,
        "is_mfg_or_services": is_mfg_or_services,
        "not_already_on_ondc": not_on_ondc,
        "delivered_orders": delivered_orders,
        "min_delivered_orders": min_orders_met,
        "overall_eligible": is_micro_or_small and is_mfg_or_services and not_on_ondc,
    }


# ── Document Verifications (OCR) ─────────────────────────

def save_document_verification(udyam_number: str, doc_type: str, extracted: dict, verification: dict = None, raw_text: str = ""):
    conn = get_db()
    conn.execute(
        "INSERT INTO document_verifications (udyam_number, document_type, extracted_data, verification_result, raw_ocr_text) VALUES (?,?,?,?,?)",
        (udyam_number, doc_type, json.dumps(extracted), json.dumps(verification or {}), raw_text),
    )
    conn.commit()
    conn.close()


def get_document_verifications(udyam_number: str) -> list:
    conn = get_db()
    rows = conn.execute(
        "SELECT * FROM document_verifications WHERE udyam_number=? ORDER BY created_at DESC",
        (udyam_number,),
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        if d.get("extracted_data"):
            d["extracted_data"] = json.loads(d["extracted_data"])
        if d.get("verification_result"):
            d["verification_result"] = json.loads(d["verification_result"])
        result.append(d)
    return result


def save_claim_verification(claim_id: int, verification_result: dict):
    """Store auto-verification result in the claim's evidence JSON."""
    conn = get_db()
    row = conn.execute("SELECT evidence FROM claims WHERE id=?", (claim_id,)).fetchone()
    if row:
        evidence = json.loads(row["evidence"]) if row["evidence"] else {}
        evidence["auto_verification"] = verification_result
        conn.execute(
            "UPDATE claims SET evidence=? WHERE id=?",
            (json.dumps(evidence), claim_id),
        )
        conn.commit()
    conn.close()
