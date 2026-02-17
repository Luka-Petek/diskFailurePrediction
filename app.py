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
    "Understanding context...",
    "Thinking longer for better answer...",
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
Zaradi lightweight llama modela, ima model v kontekstu zadnjih 20 vpraščanj uporabnika.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

# pretekla sporocila
for message in st.session_state.messages:
    # Ohranimo ikone pri izrisu zgodovine
    avatar = "🤖" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Vprašaj me karkoli..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context_data = importance_df.head(15).to_string(index=False) if importance_df is not None else ""

    system_prompt = f"""
    You are a highly knowledgeable advisor specializing in data storage reliability and disk failure prediction, 
    with expertise in interpreting results from systems based on classification, regression, and clustering. 
    Your task is to help everyday users understand how the system evaluates disk health and what actions they can take 
    to protect their data, even if they have no prior technical knowledge.

    SYSTEM SPECIFICATIONS TO PRESENT TO THE USER:
    - Overall prediction accuracy: 90.15%
    - Recall (ability to detect actual failures): 86%
    - Key SMART parameters that drive the system's decisions:
    {context_data}

    GUIDELINES FOR COMMUNICATING WITH THE USER:
    1. Focus solely on the user – all answers should help them interpret results and understand actions, not explain the model itself. 
    2. Use clear, friendly, and accessible language; avoid technical jargon unless necessary. 
    3. When explaining SMART parameters or system outputs, clarify why they matter and how they affect disk reliability, using practical examples. 
    4. If the user asks about warnings, risks, or disk issues, provide concrete, easy-to-follow preventive or corrective actions. 
    5. Include simple explanations of:
        - How SMART technology works
        - Environmental factors such as temperature and power stability affecting disk health
        - Basic maintenance tips and data protection strategies
    6. Your explanations should be objective, authoritative, and professional, but still understandable — the user should feel informed and confident about their data safety.
    7. For complex results, explain step-by-step using examples or analogies to ensure the everyday user can follow along.
    8. Always refer to “the analysis system” or “the utilized model,” never to the author or developer of the model.

    Goal: Enable the user to understand, interpret, and take informed action regarding disks and SMART parameters, 
    so they feel confident and secure using their data.
    """

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()

        # TAKOJ izpišemo naključen procesni stavek, da uporabnik vidi aktivnost
        thinking_text = random.choice(placeholder_vprasanja)
        message_placeholder.markdown(f"*{thinking_text}*")

        full_response = ""

        try:
            url = "http://ollama:11434/api/generate"

            # sliding windows, kontekst sledi samo zadnjim 25 vprasanjem..
            MAX_HISTORY = 25
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

            response = requests.post(url, json=payload, timeout=500)

            if response.status_code == 200:
                full_response = response.json().get('response', 'AI ni vrnil odgovora.')
            else:
                full_response = f"Napaka: Ollama je vrnila status {response.status_code}."

        except requests.exceptions.ConnectionError:
            full_response = "Napaka: Ne morem se povezati z Ollama storitvijo. Preveri, če container 'ollama_service' teče."
        except Exception as e:
            full_response = f"Prišlo je do napake: {e}"

        # Dejanski odgovor prepiše procesni stavek
        message_placeholder.markdown(full_response)

    # dodajanje odgovorov
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # refresh
    st.session_state.last_placeholder_update = 0
