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
        st.error(f"Error loading files: {e}")
        return None, None


importance_df, model_rf = load_resources()

st.sidebar.title("📊 Model Data")

if st.sidebar.button("🗑️ Clear chat history"):
    st.session_state.messages = []
    st.rerun()

if importance_df is not None:
    st.sidebar.write("### Feature Importance (Top 10)")
    st.sidebar.dataframe(importance_df.head(10), hide_index=True)

st.sidebar.markdown("---")
st.sidebar.write("**Accuracy:** 90.15%")
st.sidebar.write("**Recall (correctly predicted failures):** 86%")
st.sidebar.info("The model is primary based on the Random Forest algorithm, both for classification and regression and was trained on 8,828 balanced instances.")

st.title("🤖 DiskML AI Advisor")
st.markdown("""
This interface allows you to chat with an AI model about the logic of the disk failure prediction model.
You can ask anything about the model, machine learning, or disk failure influences in general.
""")

if "messages" not in st.session_state:
    st.session_state.messages = []

# pretekla sporocila
for message in st.session_state.messages:
    # Ohranimo ikone pri izrisu zgodovine
    avatar = "🤖" if message["role"] == "assistant" else None
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me anything..."):
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

        # Prikaz začetne animacije
        thinking_text = random.choice(placeholder_vprasanja)
        message_placeholder.markdown(f"*{thinking_text}*")

        try:
            url = "http://host.docker.internal:11434/api/generate"

            history_context = ""
            for msg in st.session_state.messages[:-1]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_context += f"{role}: {msg['content']}\n"

            payload = {
                "model": "diskml-mistral",
                "prompt": f"{system_prompt}\n\nHistory:\n{history_context}\nUser: {prompt}",
                "stream": True  #dodan streaming
            }

            # Pošljemo zahtevek s stream=True
            response = requests.post(url, json=payload, timeout=500, stream=True)

            if response.status_code == 200:
                full_response = ""
                # Procesiranje toka podatkov (chunk po chunk)
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line.decode('utf-8'))
                        if 'response' in chunk:
                            content = chunk.get('response', '')
                            full_response += content
                            # Sprotno osveževanje vmesnika z dodajanjem kurzorja
                            message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)
            else:
                full_response = f"Error: Ollama returned status {response.status_code}."
                message_placeholder.markdown(full_response)

        except requests.exceptions.ConnectionError:
            full_response = "Error: Cannot connect to Ollama"
            message_placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"An error occurred: {e}"
            message_placeholder.markdown(full_response)

    # dodajanje odgovorov
    st.session_state.messages.append({"role": "assistant", "content": full_response})

    # refresh
    st.session_state.last_placeholder_update = 0