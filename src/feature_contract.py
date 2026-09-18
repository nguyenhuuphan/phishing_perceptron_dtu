"""Versioned UCI feature and target contract shared by training and inference."""

SCHEMA_VERSION = "uci30-v1"

FEATURE_NAMES = [
    "having_IP_Address",
    "URL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report",
]

FEATURE_DOMAINS = {
    "having_IP_Address": {-1, 1},
    "URL_Length": {-1, 0, 1},
    "Shortining_Service": {-1, 1},
    "having_At_Symbol": {-1, 1},
    "double_slash_redirecting": {-1, 1},
    "Prefix_Suffix": {-1, 1},
    "having_Sub_Domain": {-1, 0, 1},
    "SSLfinal_State": {-1, 0, 1},
    "Domain_registeration_length": {-1, 1},
    "Favicon": {-1, 1},
    "port": {-1, 1},
    "HTTPS_token": {-1, 1},
    "Request_URL": {-1, 1},
    "URL_of_Anchor": {-1, 0, 1},
    "Links_in_tags": {-1, 0, 1},
    "SFH": {-1, 0, 1},
    "Submitting_to_email": {-1, 1},
    "Abnormal_URL": {-1, 1},
    "Redirect": {0, 1},
    "on_mouseover": {-1, 1},
    "RightClick": {-1, 1},
    "popUpWidnow": {-1, 1},
    "Iframe": {-1, 1},
    "age_of_domain": {-1, 1},
    "DNSRecord": {-1, 1},
    "web_traffic": {-1, 0, 1},
    "Page_Rank": {-1, 1},
    "Google_Index": {-1, 1},
    "Links_pointing_to_page": {-1, 0, 1},
    "Statistical_report": {-1, 1},
}

# Raw UCI Result: -1 is phishing and +1 is legitimate.  The application uses
# class 1 as the positive (phishing) class for metric and API consistency.
RAW_LABEL_TO_INTERNAL = {-1: 1, 1: 0}
INTERNAL_CLASS_NAMES = {0: "legitimate", 1: "phishing"}


def validate_feature_values(frame) -> None:
    """Reject values outside the ARFF domain while allowing missing cells."""
    for name in FEATURE_NAMES:
        observed = set(frame[name].dropna().unique())
        invalid = observed - FEATURE_DOMAINS[name]
        if invalid:
            raise ValueError(f"{name} contains values outside its domain: {sorted(invalid)}")
