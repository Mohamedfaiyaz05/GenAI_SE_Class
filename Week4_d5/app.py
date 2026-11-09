"""Streamlit interface for the medical support chatbot."""

import streamlit as st
from dotenv import load_dotenv

from flows.support_flow import respond


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Medical Support", page_icon="🩺")
    st.title("Medical Support Assistant")
    st.warning(
        "General information only—not a diagnosis or a substitute for "
        "professional medical care. For emergencies, contact your local "
        "emergency service immediately."
    )

    question = st.text_area(
        "What would you like general information about?",
        placeholder="Avoid including your name or other identifying details.",
    )
    if st.button("Get information", type="primary"):
        if not question.strip():
            st.error("Enter a question to continue.")
            return
        with st.spinner("Looking in the medical handbook..."):
            st.markdown(respond(question))


if __name__ == "__main__":
    main()
