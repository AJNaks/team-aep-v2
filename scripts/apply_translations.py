# -*- coding: utf-8 -*-
"""
apply_translations.py

Reads data/i18n_translations.json and regenerates the `const I18N = { ... };`
block inside templates/index.html (lines 427-769).

English values are kept as they appear in the ORIGINAL index.html (em-dash,
HTML span/strong templates, {text}/{type}/{value} placeholders).  For every
other language the translated strings come from the JSON, with
`<span class="mobile">{mobile}</span>` appended to otpSubtitle and
`<strong>{otp}</strong>` appended to demoHint.
"""

import json
import os
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRANSLATIONS_PATH = os.path.join(BASE_DIR, "data", "i18n_translations.json")
INDEX_HTML_PATH = os.path.join(BASE_DIR, "templates", "index.html")

# ---------------------------------------------------------------------------
# The canonical key order (matches the original I18N block in index.html)
# ---------------------------------------------------------------------------
KEY_ORDER = [
    "aiName",
    "welcome",
    "stepVerify", "stepOTP", "stepDetails",
    "tabUdyam", "tabUdyamHint",
    "tabMobile", "tabMobileHint",
    "tabPAN", "tabPANHint",
    "placeholderUdyam", "placeholderMobile", "placeholderPAN",
    "hintUdyam", "hintMobile", "hintPAN",
    "submit",
    "verifying",
    "otpTitle",
    "otpSubtitle",
    "otpBtnText",
    "otpVerifying",
    "demoHint",
    "verifiedBadge",
    "cardBusiness", "cardContact", "cardNIC",
    "continueBtn",
    "lblName", "lblType", "lblActivity", "lblOrg",
    "lblUdyam", "lblRegDate", "lblGender", "lblCategory",
    "lblMobile", "lblEmail", "lblAddress", "lblDistrict",
    "lblState", "lblPincode", "lblDIC", "lblMSMEDI",
    "notFound",
    "otpInvalid",
    "recording",
    "processing",
    "micError",
    "extracting",
    "extracted",
    "extractFail",
    "speechWelcomeShort",
    # Role selection
    "roleTitle",
    "roleSubtitle",
    "roleMsme",
    "roleMsmeDesc",
    "roleSnp",
    "roleSnpDesc",
    "roleNsic",
    "roleNsicDesc",
    # Resume
    "resumeTitle",
    "resumeMsg",
    "resumeStep",
    "resumeBtn",
    "resumeStartOver",
    "progressSaved",
    # Step labels
    "stepSNPMatch",
    "stepCatalogue",
    "stepSummary",
    # Categorisation
    "catRunning",
    "catDomain",
    "catConfidence",
    "catReasoning",
    "catContinueToSNP",
    # SNP Match
    "snpTitle",
    "snpSubtitle",
    "snpExplainer",
    "snpScore",
    "snpStrengths",
    "snpConsiderations",
    "snpSelect",
    "snpSelected",
    "snpContinueToCatalogue",
    # Catalogue builder
    "catTitle",
    "catSubtitle",
    "catAddProduct",
    "catProductName",
    "catProductDesc",
    "catProductPrice",
    "catProductUnit",
    "catSave",
    "catRemove",
    "catProducts",
    "catContinueToSummary",
    "catNoProducts",
    # Summary
    "summTitle",
    "summSubtitle",
    "summPartner",
    "summProducts",
    "summTimeline",
    "summStep1",
    "summStep1Desc",
    "summStep2",
    "summStep2Desc",
    "summStep3",
    "summStep3Desc",
    "summStep4",
    "summStep4Desc",
    "summStep5",
    "summStep5Desc",
    "summStep6",
    "summStep6Desc",
    "summBackToHome",
    # SNP Dashboard
    "snpDashTitle",
    "snpDashOverview",
    "snpDashMSEs",
    "snpDashClaims",
    "snpDashSettings",
    "snpTotalMSEs",
    "snpActiveMSEs",
    "snpPendingCatalogues",
    "snpTotalClaims",
    "snpLoginTitle",
    "snpLoginSubtitle",
    "snpLoginEmail",
    "snpLoginSubscriber",
    "snpLoginBtn",
    # NSIC Dashboard
    "nsicDashTitle",
    "nsicDashOverview",
    "nsicDashMSEs",
    "nsicDashSNPs",
    "nsicDashClaims",
    "nsicDashReports",
    "nsicLoginTitle",
    "nsicLoginSubtitle",
    "nsicLoginBtn",
    "nsicTotalMSEs",
    "nsicTotalSNPs",
    "nsicPendingClaims",
    "nsicFundsUtilized",
    # Common
    "logout",
    "enterprise",
    "ondcDomain",
    "status",
    "actions",
    "viewDetails",
    "approve",
    "reject",
    "noData",
    "loading",
]

# ---------------------------------------------------------------------------
# Keys that can share a line (compacted pairs on a single line in the
# original file).  Each tuple lists keys that go on one line together.
# ---------------------------------------------------------------------------
COMPACT_GROUPS = [
    ("tabUdyam", "tabUdyamHint"),
    ("tabMobile", "tabMobileHint"),
    ("tabPAN", "tabPANHint"),
    ("lblName", "lblType"),
    ("lblActivity", "lblOrg"),
    ("lblUdyam", "lblRegDate"),
    ("lblGender", "lblCategory"),
    ("lblMobile", "lblEmail"),
    ("lblAddress", "lblDistrict"),
    ("lblState", "lblPincode"),
    ("lblDIC", "lblMSMEDI"),
]

# Build a fast lookup: key -> group tuple it belongs to
_key_to_group = {}
for grp in COMPACT_GROUPS:
    for k in grp:
        _key_to_group[k] = grp

# ---------------------------------------------------------------------------
# Hard-coded English overrides (taken verbatim from the ORIGINAL index.html)
# These must NOT come from the translations JSON.
# ---------------------------------------------------------------------------
EN_OVERRIDES = {
    "welcome": "Welcome to ONDC registration! I'm your AI assistant. Let's start by verifying your Udyam registration. You have three options \u2014 you can provide your Udyam number, your registered mobile number, or your PAN number. Please select one of the three options below, and then either click the microphone button to speak your details, or type them into the text box.",
    "otpSubtitle": 'An OTP has been sent to your registered mobile <span class="mobile">{mobile}</span>',
    "demoHint": "Demo mode \u2014 Use OTP: <strong>{otp}</strong>",
    "extracting": 'I heard: "{text}". Let me extract the details...',
    "extracted": "I found your {type}: {value}. Verifying now...",
    "extractFail": "I couldn't identify a clear {type} from your speech. Could you try again or type it in?",
    "resumeStep": "Last step: {step}",
    "snpSelected": "\u2713 Partner Selected",
    "catProductPrice": "Price (\u20b9)",
    "catSubtitle": "Let's build your product catalogue for ONDC. Tell me about your products \u2014 you can type, speak, or add photos.",
    "summSubtitle": "Your ONDC onboarding is complete. Here's what happens next.",
    "summStep6Desc": "You're ready to receive orders from across India!",
    "roleSubtitle": "Choose how you'd like to use the platform",
    "summTitle": "You're All Set!",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _needs_double_quotes(value):
    """Return True when the JS string must use double quotes (contains ')."""
    return "'" in value


def _js_quote(value):
    """
    Return a JavaScript-quoted string.

    * If the value contains a single-quote we wrap in double quotes.
    * Otherwise we wrap in single quotes.
    * Internal double-quotes inside a double-quoted string are escaped.
    * Internal backslashes are escaped.
    """
    if _needs_double_quotes(value):
        # Escape backslashes first, then double-quotes
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return '"%s"' % escaped
    else:
        escaped = value.replace("\\", "\\\\")
        return "'%s'" % escaped


def _format_kv(key, value, indent=4):
    """Return a single `key: <quoted-value>` fragment (no trailing comma)."""
    return "%s%s: %s" % (" " * indent, key, _js_quote(value))


def _build_lang_block(lang_code, translations, is_english=False):
    """
    Build the lines for a single language inside the I18N object.
    Returns a list of strings (without trailing newlines).
    """
    lines = []
    lines.append("  %s: {" % lang_code)

    emitted = set()

    for key in KEY_ORDER:
        if key in emitted:
            continue

        # Determine the value
        if is_english and key in EN_OVERRIDES:
            val = EN_OVERRIDES[key]
        elif key in translations:
            val = translations[key]
        else:
            continue  # skip missing keys

        # For non-English: append HTML templates to otpSubtitle / demoHint
        if not is_english:
            if key == "otpSubtitle":
                val = val.rstrip()
                if '<span class="mobile">{mobile}</span>' not in val:
                    val = val + ' <span class="mobile">{mobile}</span>'
            elif key == "demoHint":
                val = val.rstrip()
                if "<strong>{otp}</strong>" not in val:
                    val = val + " <strong>{otp}</strong>"

        # Check if this key starts a compact group
        grp = _key_to_group.get(key)
        if grp and key == grp[0]:
            # Emit all keys in this group on one line
            parts = []
            for gk in grp:
                if is_english and gk in EN_OVERRIDES:
                    gv = EN_OVERRIDES[gk]
                elif gk in translations:
                    gv = translations[gk]
                else:
                    continue

                # Apply non-English appendages for group keys too
                if not is_english:
                    if gk == "otpSubtitle":
                        gv = gv.rstrip()
                        if '<span class="mobile">{mobile}</span>' not in gv:
                            gv = gv + ' <span class="mobile">{mobile}</span>'
                    elif gk == "demoHint":
                        gv = gv.rstrip()
                        if "<strong>{otp}</strong>" not in gv:
                            gv = gv + " <strong>{otp}</strong>"

                parts.append("%s: %s" % (gk, _js_quote(gv)))
                emitted.add(gk)

            lines.append("    %s," % ", ".join(parts))
        elif grp and key != grp[0]:
            # This key is part of a group but not the leader; it was already
            # emitted with the leader.
            continue
        else:
            lines.append("    %s," % ("%s: %s" % (key, _js_quote(val))))
            emitted.add(key)

    lines.append("  },")
    return lines


# ---------------------------------------------------------------------------
# Language ordering (matches the original index.html)
# ---------------------------------------------------------------------------
LANG_ORDER = ["en", "hi", "ta", "te", "bn", "mr", "gu", "kn", "ml", "pa", "od", "as"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Read translations JSON
    with open(TRANSLATIONS_PATH, "r", encoding="utf-8") as f:
        all_translations = json.load(f)

    # 2. Build the I18N JS block
    js_lines = ["const I18N = {"]

    for lang in LANG_ORDER:
        if lang not in all_translations:
            continue
        is_en = (lang == "en")
        block = _build_lang_block(lang, all_translations[lang], is_english=is_en)
        js_lines.extend(block)

    js_lines.append("};")

    i18n_block = "\n".join(js_lines) + "\n"

    # 3. Read the current index.html
    with open(INDEX_HTML_PATH, "r", encoding="utf-8") as f:
        html_lines = f.readlines()

    # 4. Locate the I18N block boundaries (1-indexed line numbers from the spec)
    #    Line 427 = start  ("const I18N = {")
    #    Line 769 = end    ("};")
    #    We'll do a safe search in case line numbers drift slightly.
    start_idx = None
    end_idx = None
    for i, line in enumerate(html_lines):
        if start_idx is None and line.strip().startswith("const I18N = {"):
            start_idx = i
        if start_idx is not None and end_idx is None:
            if line.strip() == "};":
                end_idx = i
                break

    if start_idx is None or end_idx is None:
        raise RuntimeError(
            "Could not locate the I18N block in %s. "
            "Looked for 'const I18N = {' ... '};'" % INDEX_HTML_PATH
        )

    # 5. Replace the block (start_idx through end_idx inclusive)
    new_html_lines = html_lines[:start_idx] + [i18n_block] + html_lines[end_idx + 1:]

    # 6. Write back
    with open(INDEX_HTML_PATH, "w", encoding="utf-8") as f:
        f.writelines(new_html_lines)

    print("Done. Replaced I18N block (lines %d-%d) in %s" % (
        start_idx + 1, end_idx + 1, INDEX_HTML_PATH
    ))


if __name__ == "__main__":
    main()
