# -*- coding: utf-8 -*-
"""
Generate I18N translations using Sarvam AI Translate API.
Translates all English UI strings into 11 Indian languages.

Usage: python scripts/generate_translations.py
Output: data/i18n_translations.json (ready to paste into index.html)
"""

import os, json, time, sys
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Load env from project root
PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")

SARVAM_KEY = os.getenv("SARVAM_API_KEY", "")
if not SARVAM_KEY:
    print("ERROR: SARVAM_API_KEY not set in .env")
    sys.exit(1)

# ── English source strings ──────────────────────────────────
EN_STRINGS = {
    "aiName": "AI Assistant",
    "welcome": "Welcome to ONDC registration! I'm your AI assistant. Let's start by verifying your Udyam registration. You have three options -you can provide your Udyam number, your registered mobile number, or your PAN number. Please select one of the three options below, and then either click the microphone button to speak your details, or type them into the text box.",
    "stepVerify": "Step 1 · Udyam Verification",
    "stepOTP": "Step 2 · OTP Verification",
    "stepDetails": "Step 3 · Business Details",
    "tabUdyam": "Udyam Number",
    "tabUdyamHint": "UDYAM-XX-00-0000000",
    "tabMobile": "Mobile Number",
    "tabMobileHint": "Registered mobile",
    "tabPAN": "PAN Number",
    "tabPANHint": "ABCDE1234F",
    "placeholderUdyam": "Enter Udyam number (e.g. UDYAM-MH-01-0012345)",
    "placeholderMobile": "Enter registered mobile number",
    "placeholderPAN": "Enter PAN number",
    "hintUdyam": "Format: UDYAM-XX-00-0000000",
    "hintMobile": "10-digit registered mobile",
    "hintPAN": "Format: ABCDE1234F",
    "submit": "→",
    "verifying": "Verifying...",
    "otpTitle": "Verify your identity",
    "otpSubtitle": "An OTP has been sent to your registered mobile",
    "otpBtnText": "Verify OTP",
    "otpVerifying": "Verifying...",
    "demoHint": "Demo mode -Use OTP:",
    "verifiedBadge": "✓ Udyam Verified",
    "cardBusiness": "Business Information",
    "cardContact": "Contact & Location",
    "cardNIC": "NIC Classifications",
    "continueBtn": "Continue to ONDC Mapping →",
    "lblName": "Enterprise Name",
    "lblType": "Enterprise Type",
    "lblActivity": "Major Activity",
    "lblOrg": "Organization Type",
    "lblUdyam": "Udyam Number",
    "lblRegDate": "Registration Date",
    "lblGender": "Owner Gender",
    "lblCategory": "Social Category",
    "lblMobile": "Mobile",
    "lblEmail": "Email",
    "lblAddress": "Address",
    "lblDistrict": "District",
    "lblState": "State",
    "lblPincode": "Pincode",
    "lblDIC": "DIC Office",
    "lblMSMEDI": "MSME-DI",
    "notFound": "No enterprise found. Please check and try again.",
    "otpInvalid": "Invalid OTP. Please try again.",
    "recording": "Listening...",
    "processing": "Processing speech...",
    "micError": "Microphone not available. Please type instead.",
    "extracting": "I heard your details. Let me extract the information...",
    "extracted": "I found your details. Verifying now...",
    "extractFail": "I couldn't identify clear details from your speech. Could you try again or type it in?",
    "speechWelcomeShort": "Welcome! Please verify your Udyam registration to begin ONDC onboarding.",
    # Role selection
    "roleTitle": "Select your role",
    "roleSubtitle": "Choose how you'd like to use the platform",
    "roleMsme": "I am an MSME",
    "roleMsmeDesc": "Register your enterprise on ONDC with AI assistance",
    "roleSnp": "I am an SNP",
    "roleSnpDesc": "Manage your assigned MSEs, catalogues and incentive claims",
    "roleNsic": "NSIC Official",
    "roleNsicDesc": "Verify claims, monitor MSE onboarding, and generate reports",
    # Resume
    "resumeTitle": "Welcome back!",
    "resumeMsg": "You have a saved session. Resume from where you left off?",
    "resumeStep": "Last step:",
    "resumeBtn": "Resume",
    "resumeStartOver": "Start Over",
    "progressSaved": "Progress saved",
    # Step labels
    "stepSNPMatch": "Step 4 · SNP Matching",
    "stepCatalogue": "Step 5 · Catalogue Builder",
    "stepSummary": "Summary",
    # Categorisation
    "catRunning": "Analysing your business for ONDC...",
    "catDomain": "ONDC Domain",
    "catConfidence": "Confidence",
    "catReasoning": "Why this domain?",
    "catContinueToSNP": "Continue to SNP Matching →",
    # SNP Match
    "snpTitle": "Choose Your ONDC Partner",
    "snpSubtitle": "Based on your business profile, we've found the best Seller Network Participants for you",
    "snpExplainer": "An SNP (Seller Network Participant) helps you get listed and start selling on the ONDC network. They handle your digital catalogue, logistics integration, and order management.",
    "snpScore": "Match Score",
    "snpStrengths": "Strengths",
    "snpConsiderations": "Consider",
    "snpSelect": "Select This Partner",
    "snpSelected": "Partner Selected",
    "snpContinueToCatalogue": "Continue to Catalogue Builder →",
    # Catalogue builder
    "catTitle": "AI Catalogue Builder",
    "catSubtitle": "Let's build your product catalogue for ONDC. Tell me about your products — you can type, speak, or add photos.",
    "catAddProduct": "Add Product",
    "catProductName": "Product Name",
    "catProductDesc": "Description",
    "catProductPrice": "Price",
    "catProductUnit": "Unit",
    "catSave": "Save Product",
    "catRemove": "Remove",
    "catProducts": "Products Added",
    "catContinueToSummary": "Continue to Summary →",
    "catNoProducts": "No products added yet. Add your first product above.",
    # Summary
    "summTitle": "You're All Set!",
    "summSubtitle": "Your ONDC onboarding is complete. Here's what happens next.",
    "summPartner": "Your SNP Partner",
    "summProducts": "Products in Catalogue",
    "summTimeline": "What Happens Next",
    "summStep1": "Catalogue Review",
    "summStep1Desc": "Your SNP partner will review and enhance your product catalogue",
    "summStep2": "ONDC Listing",
    "summStep2Desc": "Your products go live on the ONDC network",
    "summStep3": "Logistics Setup",
    "summStep3Desc": "Delivery and fulfilment partners are configured",
    "summStep4": "Payment Integration",
    "summStep4Desc": "Secure payment gateway setup for receiving orders",
    "summStep5": "Training",
    "summStep5Desc": "Learn to manage orders, inventory and customer queries",
    "summStep6": "Start Selling",
    "summStep6Desc": "You're ready to receive orders from across India!",
    "summBackToHome": "Back to Home",
    # SNP Dashboard
    "snpDashTitle": "SNP Dashboard",
    "snpDashOverview": "Overview",
    "snpDashMSEs": "My MSEs",
    "snpDashClaims": "Claims",
    "snpDashSettings": "Settings",
    "snpTotalMSEs": "Total MSEs",
    "snpActiveMSEs": "Active MSEs",
    "snpPendingCatalogues": "Pending Catalogues",
    "snpTotalClaims": "Total Claims",
    "snpLoginTitle": "SNP Login",
    "snpLoginSubtitle": "Sign in to manage your MSEs and claims",
    "snpLoginEmail": "Email Address",
    "snpLoginSubscriber": "ONDC Subscriber ID",
    "snpLoginBtn": "Send OTP",
    # NSIC Dashboard
    "nsicDashTitle": "NSIC Dashboard",
    "nsicDashOverview": "Overview",
    "nsicDashMSEs": "MSE Registry",
    "nsicDashSNPs": "SNP Registry",
    "nsicDashClaims": "Claims",
    "nsicDashReports": "Reports",
    "nsicLoginTitle": "NSIC Login",
    "nsicLoginSubtitle": "Sign in with your Employee ID",
    "nsicLoginBtn": "Send OTP",
    "nsicTotalMSEs": "Total MSEs Onboarded",
    "nsicTotalSNPs": "Active SNPs",
    "nsicPendingClaims": "Pending Claims",
    "nsicFundsUtilized": "Funds Utilized",
    # Common
    "logout": "Logout",
    "enterprise": "Enterprise",
    "ondcDomain": "ONDC Domain",
    "status": "Status",
    "actions": "Actions",
    "viewDetails": "View Details",
    "approve": "Approve",
    "reject": "Reject",
    "noData": "No data available",
    "loading": "Loading...",
    # TEAM Registration ID
    "teamRegIdLabel": "Your TEAM Registration ID",
    "teamRegIdHint": "Save this ID. You will need it for all future TEAM correspondence.",
    "copyId": "Copy ID",
    "readAloud": "Read aloud",
    "idCopied": "Registration ID copied!",
    "idReadAloud": "Reading your registration ID...",
    "teamRegIdSpoken": "Your TEAM registration ID is",
    # Voice journey UI
    "vjPhaseReg": "Registration",
    "vjPhaseDocs": "Documents",
    "vjPhaseCat": "Catalogue",
    "vjPhaseSnp": "SNP Match",
    "vjVerifying": "Verifying your Udyam number...",
    "vjOtpSent": "Verified! OTP sent to number ending in {digits}. Please tell me the six digit code.",
    "vjVerifyFailed": "Could not verify that number. Let me ask for your mobile number instead.",
    "vjNetworkError": "Network error. Let's try your mobile number instead.",
    "vjTapToSpeak": "Tap to speak",
    "vjListening": "Listening... tap to stop",
    "vjDidntCatch": "Sorry, I didn't catch that. Please try again.",
}

# Keys that should NOT be translated (they are codes, symbols, or formats)
SKIP_KEYS = {
    "tabUdyamHint",    # UDYAM-XX-00-0000000 -a format code
    "tabPANHint",      # ABCDE1234F -a format code
    "hintUdyam",       # Format: UDYAM-XX-00-0000000
    "hintPAN",         # Format: ABCDE1234F
    "submit",          # → (arrow symbol)
    "snpSelected",     # contains ✓ symbol prefix
}

# Target languages (Sarvam BCP-47 codes)
TARGET_LANGUAGES = {
    "hi": "hi-IN",
    "ta": "ta-IN",
    "te": "te-IN",
    "bn": "bn-IN",
    "mr": "mr-IN",
    "gu": "gu-IN",
    "kn": "kn-IN",
    "ml": "ml-IN",
    "pa": "pa-IN",
    "od": "od-IN",
    "as": "as-IN",
}

LANG_NAMES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali",
    "mr": "Marathi", "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam",
    "pa": "Punjabi", "od": "Odia", "as": "Assamese",
}


def translate_text(text: str, target_lang_code: str, model: str = "mayura:v1") -> str:
    """Translate a single string from English to target language using Sarvam API."""
    try:
        with httpx.Client(timeout=30) as client:
            resp = client.post(
                "https://api.sarvam.ai/translate",
                headers={
                    "Content-Type": "application/json",
                    "api-subscription-key": SARVAM_KEY,
                },
                json={
                    "input": text,
                    "source_language_code": "en-IN",
                    "target_language_code": target_lang_code,
                    "model": model,
                    "mode": "formal",
                    "enable_preprocessing": True,
                },
            )
            data = resp.json()
            translated = data.get("translated_text", "")
            if translated:
                return translated
            else:
                print(f"    WARNING: Empty translation returned. Response: {data}")
                return text
    except Exception as e:
        print(f"    ERROR translating: {e}")
        return text


def main():
    output_path = PROJECT_ROOT / "data" / "i18n_translations.json"

    # Load existing translations if available (to resume interrupted runs)
    all_translations = {"en": EN_STRINGS}
    if output_path.exists():
        with open(output_path) as f:
            all_translations = json.load(f)
        print(f"Loaded existing translations from {output_path}")

    translatable_keys = [k for k in EN_STRINGS if k not in SKIP_KEYS]
    total_calls = len(TARGET_LANGUAGES) * len(translatable_keys)
    completed = 0

    print(f"\nTranslating {len(translatable_keys)} strings × {len(TARGET_LANGUAGES)} languages = {total_calls} API calls")
    print(f"Skipping keys: {SKIP_KEYS}\n")

    for lang_code, sarvam_code in TARGET_LANGUAGES.items():
        lang_name = LANG_NAMES[lang_code]
        print(f"═══ Translating to {lang_name} ({sarvam_code}) ═══")

        if lang_code not in all_translations:
            all_translations[lang_code] = {}

        lang_data = all_translations[lang_code]

        for key in EN_STRINGS:
            if key in SKIP_KEYS:
                # Copy as-is (format codes, symbols)
                lang_data[key] = EN_STRINGS[key]
                continue

            # Skip if already translated in a previous run
            if key in lang_data and lang_data[key] != EN_STRINGS[key] and lang_data[key]:
                completed += 1
                continue

            en_text = EN_STRINGS[key]
            print(f"  [{completed+1}/{total_calls}] {key}: \"{en_text[:50]}...\"" if len(en_text) > 50 else f"  [{completed+1}/{total_calls}] {key}: \"{en_text}\"")

            translated = translate_text(en_text, sarvam_code)
            lang_data[key] = translated
            completed += 1

            # Brief pause to respect rate limits
            time.sleep(0.3)

        all_translations[lang_code] = lang_data

        # Save after each language (in case of interruption)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_translations, f, ensure_ascii=False, indent=2)
        print(f"  ✓ Saved {lang_name} translations\n")

    print(f"\n{'='*60}")
    print(f"DONE! All translations saved to: {output_path}")
    print(f"Total API calls made: {completed}")
    print(f"\nNext step: Run 'python scripts/apply_translations.py' to update index.html")


if __name__ == "__main__":
    main()
