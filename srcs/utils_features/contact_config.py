"""Central public destinations for the Contact / Request Analysis page."""

from email.utils import parseaddr
from urllib.parse import quote, urlencode, urlparse


# Replace these two placeholder values with the public contact destinations.
GOOGLE_FORM_URL = "https://forms.gle/DYVXmWtLTTDMwPTq7"
CONTACT_EMAIL = "sabyasachi.chakrabarty@med.ovgu.de"

EMAIL_SUBJECT = "Streamlit Application – Feedback / Analysis Request"
EMAIL_BODY = """Hello,

Request type: [Analysis request / Feature request / Feedback / Bug report / General inquiry]

Description:
[Please describe your request, suggestion, or issue here.]

Thank you.
"""


def get_configured_google_form_url():
    """Return the form URL only when the placeholder has been replaced."""
    value = GOOGLE_FORM_URL.strip()
    parsed = urlparse(value)
    if (
        not value
        or "YOUR_FORM_ID" in value
        or parsed.scheme not in {"http", "https"}
        or not parsed.netloc
    ):
        return None
    return value


def get_configured_contact_email():
    """Return a simple validated address when it is no longer a placeholder."""
    value = CONTACT_EMAIL.strip()
    _, parsed_address = parseaddr(value)
    if (
        not value
        or value == "your-email@example.com"
        or parsed_address != value
        or "@" not in parsed_address
    ):
        return None
    return value


def build_mailto_url(contact_email=None):
    """Build a safely encoded mailto link with an editable message template."""
    recipient = contact_email or get_configured_contact_email()
    if not recipient:
        return None
    query = urlencode({
        "subject": EMAIL_SUBJECT,
        "body": EMAIL_BODY,
    })
    return f"mailto:{quote(recipient, safe='@+')}?{query}"
