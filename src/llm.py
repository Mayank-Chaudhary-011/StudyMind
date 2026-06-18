import os
import streamlit as st
import google.generativeai as genai
from groq import Groq


class GeminiLimitExceeded(Exception):
    """Raised when the Gemini API rate limit or quota is exceeded."""
    pass


def get_llm_client():
    """Gets the configured LLM provider, API key, and model."""
    provider = st.session_state.get("llm_provider", "Gemini")
    model = st.session_state.get("llm_model", "gemini-2.5-flash")
    
    # Try session state first
    api_key = st.session_state.get("llm_api_key", "").strip()
    
    # Then try environment variables
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
    
    # Then try Streamlit secrets (may not exist on all deployments)
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY", "") or st.secrets.get("GOOGLE_API_KEY", "")
        except Exception:
            api_key = ""
            
    return provider, api_key, model

def generate_completion(prompt, temperature=0.1, max_tokens=1024):
    """Generates completion using either Groq or Google Gemini based on settings."""
    provider, api_key, model = get_llm_client()
    
    if not api_key:
        raise ValueError(f"API Key for {provider} is not configured. Please set it in the settings panel or environment.")

    if provider == "Groq":
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    elif provider == "Gemini":
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        client = genai.GenerativeModel(model_name=model)
        try:
            response = client.generate_content(prompt, generation_config=generation_config)
            return response.text
        except Exception as e:
            error_msg = str(e).lower()
            if any(keyword in error_msg for keyword in [
                "429", "resource_exhausted", "rate limit",
                "quota", "too many requests", "resourceexhausted"
            ]):
                raise GeminiLimitExceeded(
                    "⚠️ Gemini API rate limit exceeded! "
                    "You have hit the free-tier usage limit. "
                    "Please switch to **Groq** LLM provider from the settings above to continue using the app."
                )
            raise
    else:
        raise ValueError(f"Unknown LLM Provider: {provider}")
