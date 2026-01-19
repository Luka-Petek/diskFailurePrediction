import streamlit as st
import joblib
import pandas as pd
import requests
import json

# Nastavitev strani
st.set_page_config(page_title="DiskML Analitik", layout="wide")

# Funkcija za nalaganje metapodatkov modela
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

# Stranska vrstica s podatki o modelu
st.sidebar.title("📊 Podatki o modelu")
if importance_df is not None:
    st.sidebar.write("### Pomembnost značilk (Top 10)")
    st.sidebar.dataframe(importance_df.head(10), hide_index=True)

st.sidebar.markdown("---")
st.sidebar.write("**Natančnost:** 90.15%")
st.sidebar.write("**Recall (Ulov odpovedi):** 86%")
st.sidebar.info("Model temelji na Random Forest algoritmu in je bil naučen na 8.828 uravnoteženih instancah.")

# Glavni del vmesnika
st.title("🤖 DiskML AI Sogovornik")
st.markdown("""
Ta vmesnik ti omogoča pogovor z AI modelom o logiki tvojega Random Forest modela za napovedovanje odpovedi diskov.
Vprašaj ga karkoli o tem, kako se model odloča.
""")

# Inicializacija zgodovine chata
if "messages" not in st.session_state:
    st.session_state.messages = []

# Prikaz preteklih sporočil
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Vnos uporabnika
if prompt := st.chat_input("Npr.: Zakaj je smart_5 tako pomemben?"):
    # Dodaj uporabnikovo vprašanje v zgodovino
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Priprava konteksta za Ollamo
    # AI-ju podamo "znanje" iz tvojega CSV-ja, da bo vedel odgovarjati specifično o tvojem modelu
    context_data = importance_df.head(15).to_string(index=False)

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
    """

    # Klic Ollama API-ja znotraj Docker omrežja
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Povezava do ollama containerja (ime 'ollama' je iz docker-compose)
            url = "http://ollama:11434/api/generate"
            payload = {
                "model": "llama3",
                "prompt": f"{system_prompt}\n\nUporabnik sprašuje: {prompt}",
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

    # Dodaj odgovor v zgodovino
    st.session_state.messages.append({"role": "assistant", "content": full_response})