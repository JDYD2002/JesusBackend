from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from ai21 import AI21Client
import requests
from gtts import gTTS
import tempfile
import base64
import os

# === Variáveis de ambiente ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI21_API_KEY = os.getenv("AI21_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_ai21 = AI21Client(api_key=AI21_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Mensagem(BaseModel):
    texto: str

# === Funções de chat ===

def chat_openai(texto):
    response = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": texto}]
    )
    return response.choices[0].message.content.strip()

def chat_ai21(texto):
    response = client_ai21.generate_text(
        prompt=texto,
        maxTokens=200,
        temperature=0.7,
        topP=0.9
    )
    return response.text.strip()

def chat_fireworks(mensagem_texto):
    url = "https://api.fireworksai.com/v1/generate"
    headers = {
        "Authorization": f"Bearer {FIREWORKS_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "prompt": mensagem_texto,
        "max_tokens": 200,
        "temperature": 0.8
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json().get("text", "").strip()

def chat_groq(mensagem_texto):
    url = "https://api.groq.ai/v1/infer"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": mensagem_texto,
        "model": "default"
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json().get("output", "").strip()

def chat_hf(texto):
    # Placeholder Hugging Face - ajuste para seu endpoint e autenticação
    url = "https://api-inference.huggingface.co/models/your-model"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {"inputs": texto}
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    # Supondo que resposta é uma lista de dicts
    result = response.json()
    if isinstance(result, list) and len(result) > 0:
        return result[0].get("generated_text", "").strip()
    return ""

def chat_together(texto):
    # Placeholder Together API - ajuste conforme documentação
    url = "https://api.together.xyz/v3/chat/completions"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    payload = {
        "model": "together-gpt",
        "messages": [{"role": "user", "content": texto}]
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()

# === Endpoint chat com fallback ===
@app.post("/chat")
async def chat(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    funcoes = [
        chat_openai,
        chat_fireworks,
        chat_groq,
        chat_hf,
        chat_ai21,
        chat_together
    ]
    for func in funcoes:
        try:
            resposta = func(texto_usuario)
            if resposta:
                return {"resposta": resposta}
        except Exception as e:
            print(f"Erro {func.__name__}: {e}")
    return {"resposta": "Desculpe, Jesusinho está com dificuldade para responder agora. 🙏"}

# === TTS (áudio base64) ===
@app.post("/tts")
async def tts(mensagem: Mensagem):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            tts = gTTS(text=mensagem.texto, lang="pt-br")
            tts.save(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                audio_bytes = f.read()
            os.remove(tmpfile.name)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return {"audio_b64": audio_b64}
    except Exception as e:
        return {"audio_b64": None, "erro": str(e)}

# === Versículo do Dia ===
@app.get("/versiculo")
async def versiculo():
    prompt = "Me dê um versículo bíblico inspirador para hoje."
    try:
        resposta = chat_openai(prompt)
        return {"resposta": resposta}
    except Exception as e:
        print(f"Erro versiculo: {e}")
        return {"resposta": "Erro ao obter versículo. 🙏"}

# === Oração do Dia ===
@app.get("/oracao")
async def oracao():
    prompt = "Escreva uma oração curta e edificante para o dia de hoje."
    try:
        resposta = chat_openai(prompt)
        return {"resposta": resposta}
    except Exception as e:
        print(f"Erro oracao: {e}")
        return {"resposta": "Erro ao obter oração. 🙏"}

# === Status ===
@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando com fallback inteligente! 🙌"}
