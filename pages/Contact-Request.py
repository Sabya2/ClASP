"""Public contact options without server-side collection or email delivery."""

import streamlit as st

from srcs.utils_features.contact_config import (
    build_mailto_url,
    get_configured_contact_email,
    get_configured_google_form_url,
)


google_form_url = get_configured_google_form_url()
contact_email = get_configured_contact_email()
mailto_url = build_mailto_url(contact_email) if contact_email else None

st.title("Contact / Request Analysis")
st.markdown(
    "Have a question, an idea for a new analysis, or feedback about the "
    "application? You can contact us using either of the options below."
)

form_col, email_col = st.columns(2, gap="large")

with form_col:
    with st.container(border=True):
        st.markdown("### 📄 Submit a Feedback / Analysis Request")
        st.markdown(
            "Use our feedback form to request analyses, suggest improvements, "
            "or report issues."
        )
        if google_form_url:
            st.link_button(
                "Open Feedback Form",
                google_form_url,
                icon="📄",
                type="primary",
                width="stretch",
            )
            st.caption("The feedback form opens in a new browser tab.")
        else:
            st.info(
                "The Google feedback form has not been configured yet. Please "
                "use the email option or contact the application administrator."
            )

with email_col:
    with st.container(border=True):
        st.markdown("### ✉️ Contact via Email")
        st.markdown(
            "Prefer email? Send us your questions, suggestions, or analysis "
            "requests directly."
        )
        if mailto_url:
            st.link_button(
                "Send an Email",
                mailto_url,
                icon="✉️",
                type="primary",
                width="stretch",
            )
            st.caption(
                "This opens your configured email application with an editable "
                "message. Review it and send the email from your own account."
            )
        else:
            st.info(
                "The contact email address has not been configured yet. Please "
                "use the feedback form or contact the application administrator."
            )

st.divider()
st.caption(
    "CLASP does not send email, store your contact details, or require email "
    "credentials. Google Forms handles form submissions, and email is sent only "
    "when you choose to send it from your own email application. Do not include "
    "confidential or identifiable participant data."
)
