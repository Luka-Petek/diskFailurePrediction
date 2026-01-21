import streamlit as st
import joblib
import pandas as pd
import requests
import json
import os
import random
import time

st.set_page_config(page_title="DiskML Model", layout="wide")

if "last_placeholder_update" not in st.session_state:
    st.session_state.last_placeholder_update = 0
if "current_placeholder" not in st.session_state:
    st.session_state.current_placeholder = "Ready to help..."

placeholder_vprasanja = [
    "Processing your request...",
    "Thinking...",
    "Understanding context...",
    "Thinking longer for better answer...",
    "Performing algorithmic analysis...",
    "Synthesizing information...",
    "Formulating a response...",
    "Analyzing logic...",
    "Connecting the dots..."
]

trenutni_cas = time.time()
if trenutni_cas - st.session_state.last_placeholder_update > 10:
    st.session_state.current_placeholder = random.choice(placeholder_vprasanja)
    st.session_state.last_placeholder_update = trenutni_cas


@st.cache_resource
def load_resources():
    try:
        importance = pd.read_csv('feature_importance.csv')
        model = joblib.load('disk_model.pkl')
        return importance, model
    except Exception as e:
        st.error(f"Napaka pri nalaganju datotek: {e}")
        return None, None


importance_df, model_rf = load_resources()

st.sidebar.title("📊 Podatki o modelu")

if st.sidebar.button("🗑️ Počisti zgodovino pogovora"):
    st.session_state.messages = []
    st.rerun()

if importance_df is not None:
    st.sidebar.write("### Pomembnost značilnic (Top 10)")
    st.sidebar.dataframe(importance_df.head(10), hide_index=True)

st.sidebar.markdown("---")
st.sidebar.write("**Natančnost:** 90.15%")
st.sidebar.write("**Recall (pravilno napovedane odpovedi):** 86%")
st.sidebar.info("Model temelji na Random Forest algoritmu in je bil naučen na 8.828 uravnoteženih instancah.")

st.title("🤖 DiskML AI Sogovornik")
st.markdown("""
Ta vmesnik ti omogoča pogovor z AI modelom o logiki modela za napovedovanje odpovedi diskov.
Vprašaš ga lahko karkoli o modelu, strojnem učenju ali o vplivih na odpoved diska nasploh.

Zaradi lightweight llama modela, ima model v kontekstu zadnjih 20 vprašanj uporabnika.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

# pretekla sporocila
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input(st.session_state.current_placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_data = importance_df.head(15).to_string(index=False) if importance_df is not None else ""

    system_prompt = f"""
    Si strokovnjak za strojno učenje in shranjevanje podatkov. 
    Analiziraš moj specifičen Random Forest model za napovedovanje odpovedi diskov.

    TUKAJ SO PODATKI O MOJEM MODELU:
    - Skupna natančnost: 90.15%
    - Recall (ulov dejanskih odpovedi): 86%
    - Najpomembnejši SMART parametri (Feature Importance):
    {context_data}

    Tvoji odgovori morajo temeljiti na teh podatkih. Če te uporabnik vpraša o pomembnosti, 
    poglej v zgornji seznam. Govori strokovno, a razumljivo. 
    Lahko pričakuješ tudi splošna vprašanja o diskih nasploh, SMART skeniranju diskov, znanji vplivi na delovanje diskov in nasplošno karkoli glede te tematike in strojnem učenju.
    """

    # Klic Ollama API-ja znotraj Docker omrežja
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            url = "http://ollama:11434/api/generate"

            #sliding windows, kontekst sledi samo zadnjim 20 vprasanjem..
            MAX_HISTORY = 20
            recent_messages = st.session_state.messages[-MAX_HISTORY:]

            history_context = ""
            for msg in recent_messages[:-1]:  # vzamemo vse razen čisto zadnjega prompta
                role = "Uporabnik" if msg["role"] == "user" else "AI"
                history_context += f"{role}: {msg['content']}\n"

            payload = {
                "model": "llama3",
                "prompt": f"{system_prompt}\n\nZgodovina pogovora:\n{history_context}\nUporabnik sprašuje: {prompt}",
                "stream": False
            }

            response = requests.post(url, json=payload, timeout=120)

            if response.status_code == 200:
                full_response = response.json().get('response', 'AI ni vrnil odgovora.')
            else:
                full_response = f"Napaka: Ollama je vrnila status {response.status_code}."

        except requests.exceptions.ConnectionError:
            full_response = "Napaka: Ne morem se povezati z Ollama storitvijo. Preveri, če container 'ollama_service' teče."
        except Exception as e:
            full_response = f"Prišlo je do napake: {e}"

        message_placeholder.markdown(full_response)

    #dodajanje odgovorov
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    #refresh
    st.session_state.last_placeholder_update = 0