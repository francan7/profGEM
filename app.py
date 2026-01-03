import streamlit as st
import os
from datetime import datetime
import google.generativeai as genai

# Configurazione della pagina
st.set_page_config(
    page_title="Chatbot Profilazione",
    page_icon="🤖",
    layout="centered"
)

# Titolo dell'app
st.title("🤖 Chatbot di Profilazione")
st.markdown("*Il chatbot ti farà alcune domande per capire il tuo profilo*")

# Configurazione del modello (usa secrets per sicurezza)
@st.cache_resource
def setup_model():
    """Configura il modello Gemini."""
    # Prova a leggere dai secrets, altrimenti usa variabile d'ambiente
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except (FileNotFoundError, KeyError):
        # Fallback per test locale con secrets.toml
        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            st.error("⚠️ API Key non configurata! Assicurati di aver creato .streamlit/secrets.toml")
            st.stop()
    
    genai.configure(api_key=api_key)
    
    # Carica il system prompt
    try:
        with open("prompt.txt", "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    except FileNotFoundError:
        system_prompt = "Il tuo obiettivo è profilare l'utente con cui stai parlando. I profili possibili sono: A) il tipo scherzoso; B) il tipo serio. Fai qualche domanda per capire con che tipo di utente stai parlando"
    
    # Configurazione dei filtri di sicurezza
    safety_settings = {
        "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
        "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
        "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE",
    }
    
    model = genai.GenerativeModel(
        model_name="gemini-3-pro-preview",
        system_instruction=system_prompt,
        safety_settings=safety_settings
    )
    
    return model

# Inizializza il modello
model = setup_model()

# Inizializza lo stato della sessione
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.chat = None

if "conversation_started" not in st.session_state:
    st.session_state.conversation_started = False

# Funzione per ottenere risposta
def get_response(prompt):
    """Ottieni risposta da Gemini."""
    # Costruisci la storia
    chat_history = []
    for msg in st.session_state.messages:
        role = "user" if msg["role"] == "user" else "model"
        chat_history.append({"role": role, "parts": [msg["content"]]})
    
    # Crea chat session
    chat = model.start_chat(history=chat_history)
    
    FINISH_REASONS = {
        0: "FINISH_REASON_UNSPECIFIED",
        1: "STOP",
        2: "MAX_TOKENS",
        3: "SAFETY",
        4: "RECITATION",
        5: "OTHER"
    }
    
    try:
        response = chat.send_message(prompt)
        
        if not response.candidates:
            return "⚠️ Risposta bloccata dai filtri di sicurezza. Prova a riformulare."
        
        candidate = response.candidates[0]
        finish_reason_code = candidate.finish_reason
        
        if finish_reason_code == 3:
            return "⚠️ Risposta bloccata per motivi di sicurezza. Prova con una domanda diversa."
        
        return response.text
    
    except Exception as e:
        return f"❌ Errore: {str(e)}"

# Funzione per salvare il log
def save_conversation_log():
    """Salva la conversazione in un file."""
    if not os.path.exists("logs"):
        os.makedirs("logs")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"logs/conversation_{timestamp}.txt"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"=== Conversazione del {datetime.now().strftime('%d/%m/%Y %H:%M:%S')} ===\n")
        f.write(f"Modello: gemini-3-pro-preview\n")
        f.write(f"Provider: Google Gemini\n\n")
        for msg in st.session_state.messages:
            f.write(f"[{msg.get('timestamp', 'N/A')}] {msg['role'].upper()}: {msg['content']}\n")
        f.write(f"\n=== Fine conversazione ===\n")
    
    return filename

# Sidebar con info e controlli
with st.sidebar:
    st.header("ℹ️ Informazioni")
    st.markdown(f"**Messaggi:** {len(st.session_state.messages)}")
    st.markdown(f"**Modello:** Gemini 3.0 Pro")
    
    st.divider()
    
    if st.button("🔄 Nuova Conversazione"):
        st.session_state.messages = []
        st.session_state.chat = None
        st.session_state.conversation_started = False
        st.rerun()
    
    if st.button("💾 Salva Conversazione") and st.session_state.messages:
        filename = save_conversation_log()
        st.success(f"✅ Salvato in: {filename}")
    
    st.divider()
    st.markdown("### 📊 Informazioni Sistema")
    st.markdown("Questo chatbot profila gli utenti attraverso domande mirate.")

# Area della chat
st.divider()

# Mostra i messaggi precedenti
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input dell'utente
if prompt := st.chat_input("Scrivi il tuo messaggio..."):
    # Aggiungi messaggio utente
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Mostra messaggio utente
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Ottieni e mostra risposta
    with st.chat_message("assistant"):
        with st.spinner("Elaborazione..."):
            response = get_response(prompt)
        st.markdown(response)
    
    # Aggiungi risposta alla storia
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "timestamp": timestamp
    })
    
    st.session_state.conversation_started = True

# Messaggio iniziale se la conversazione non è iniziata
if not st.session_state.conversation_started and not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown("👋 Ciao! Sono qui per conoscerti meglio. Inizia a scrivermi un messaggio!")