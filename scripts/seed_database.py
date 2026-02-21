"""
Seed the TEAM-AEP SQLite database with mock data.
Run: python scripts/seed_database.py
"""

import sys, json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.database import init_db, get_db

DATA_DIR = Path(__file__).parent.parent / "data"


def seed():
    print("Initializing database...")
    init_db()

    conn = get_db()

    # -- Seed SNP profiles --------------------------------
    with open(DATA_DIR / "snp_profiles.json") as f:
        snp_data = json.load(f)

    for snp in snp_data["snps"]:
        profile_fields = {k: v for k, v in snp.items() if k not in ("snp_id", "name", "email", "subscriber_id", "mobile")}
        conn.execute(
            "INSERT OR REPLACE INTO snp_profiles (snp_id, name, email, subscriber_id, mobile, profile_data) VALUES (?,?,?,?,?,?)",
            (snp["snp_id"], snp["name"], snp.get("email"), snp.get("subscriber_id"), snp.get("mobile"), json.dumps(profile_fields)),
        )
    print(f"  Seeded {len(snp_data['snps'])} SNP profiles")

    # -- Seed NSIC officials ------------------------------
    with open(DATA_DIR / "mock_nsic_officials.json") as f:
        nsic_data = json.load(f)

    for official in nsic_data["officials"]:
        conn.execute(
            "INSERT OR REPLACE INTO nsic_officials (employee_id, name, mobile, role) VALUES (?,?,?,?)",
            (official["employee_id"], official["name"], official["mobile"], official["role"]),
        )
    print(f"  Seeded {len(nsic_data['officials'])} NSIC officials")

    # -- Seed MSME sessions from mock_udyam --------------─
    with open(DATA_DIR / "mock_udyam.json") as f:
        udyam_data = json.load(f)

    with open(DATA_DIR / "mock_assignments.json") as f:
        assign_data = json.load(f)

    # Build a status lookup from assignments
    status_map = {a["udyam_number"]: a["status"] for a in assign_data["assignments"]}

    for udyam_number, ent in udyam_data["enterprises"].items():
        status = status_map.get(udyam_number, "verify")
        # Map assignment status to MSME step
        step_map = {
            "assigned": "details",
            "onboarding": "details",
            "catalogue_pending": "snp_match",
            "catalogue_review": "catalogue",
            "live": "summary",
        }
        current_step = step_map.get(status, "verify")
        catalogue = assign_data.get("catalogues", {}).get(udyam_number)

        conn.execute(
            """INSERT OR REPLACE INTO msme_sessions
               (udyam_number, current_step, language, enterprise_data, catalogue_data, completed_at, updated_at)
               VALUES (?,?,?,?,?,?,datetime('now'))""",
            (
                udyam_number,
                current_step,
                "en",
                json.dumps(ent),
                json.dumps(catalogue) if catalogue else None,
                "2026-02-15T10:00:00" if status == "live" else None,
            ),
        )
    print(f"  Seeded {len(udyam_data['enterprises'])} MSME sessions")

    # -- Seed MSE-SNP assignments ------------------------─
    for assignment in assign_data["assignments"]:
        conn.execute(
            "INSERT OR REPLACE INTO mse_snp_assignments (udyam_number, snp_id, status, assigned_at) VALUES (?,?,?,?)",
            (assignment["udyam_number"], assignment["snp_id"], assignment["status"], assignment["assigned_at"]),
        )
    print(f"  Seeded {len(assign_data['assignments'])} MSE-SNP assignments")

    # -- Seed claims --------------------------------------
    for claim in assign_data["claims"]:
        conn.execute(
            """INSERT INTO claims (snp_id, udyam_number, claim_type, amount, sku_count, status, reviewed_by, review_notes, evidence)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (
                claim["snp_id"],
                claim["udyam_number"],
                claim["claim_type"],
                claim["amount"],
                claim.get("sku_count", 0),
                claim["status"],
                claim.get("reviewed_by"),
                claim.get("review_notes"),
                json.dumps(claim.get("evidence", {})),
            ),
        )
    print(f"  Seeded {len(assign_data['claims'])} claims")

    conn.commit()
    conn.close()
    print(f"\nDatabase seeded at: {DATA_DIR / 'team_aep.db'}")


if __name__ == "__main__":
    # Remove existing DB to start fresh
    db_path = DATA_DIR / "team_aep.db"
    if db_path.exists():
        db_path.unlink()
        print(f"Removed existing {db_path}")
    seed()
