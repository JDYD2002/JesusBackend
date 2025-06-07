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
HF_API_KEY = os.getenv("HF_API_KEY")
AI21_API_KEY = os.getenv("AI21_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_ai21 = AI21Client(api_key=AI21_API_KEY)

# === Função Fireworks AI ===
def chat_fireworks(mensagem_texto):
    url = "https://api.fireworksai.com/v1/generate"  # ajuste se for outro endpoint
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

# === Função Groq ===
def chat_groq(mensagem_texto):
    url = "https://api.groq.ai/v1/infer"  # ajuste se for outro endpoint oficial
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "input": mensagem_texto,
        "model": "default"  # ou o modelo que você usar
    }
    response = requests.post(url, json=payload, headers=headers, timeout=30)
    response.raise_for_status()
    return response.json().get("output", "").strip()

# Seu código de FastAPI (com middleware e modelos já definidos)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Mensagem(BaseModel):
    texto: str

# Seu histórico e outras funções (DeepSeek, OpenAI, AI21, HuggingFace, Together) aqui...

@app.post("/chat")
async def chat(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    # Tentar em sequência: DeepSeek > OpenAI > Fireworks > Groq > HF > AI21
    try:
        resposta = chat_deepseek(texto_usuario)
        return {"resposta": resposta}
    except Exception as e1:
        print(f"Erro DeepSeek: {e1}")
        try:
            resposta = chat_openai(texto_usuario)
            return {"resposta": resposta}
        except Exception as e2:
            print(f"Erro OpenAI: {e2}")
            try:
                resposta = chat_fireworks(texto_usuario)
                return {"resposta": resposta}
            except Exception as e3:
                print(f"Erro Fireworks: {e3}")
                try:
                    resposta = chat_groq(texto_usuario)
                    return {"resposta": resposta}
                except Exception as e4:
                    print(f"Erro Groq: {e4}")
                    try:
                        resposta = chat_hf(texto_usuario)
                        return {"resposta": resposta}
                    except Exception as e5:
                        print(f"Erro Hugging Face: {e5}")
                        try:
                            resposta = chat_ai21(texto_usuario)
                            return {"resposta": resposta}
                        except Exception as e6:
                            print(f"Erro AI21: {e6}")
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
    try:
        resposta = chat_deepseek("Me dê um versículo bíblico inspirador para hoje.")
        return {"resposta": resposta}
    except:
        return {"resposta": "Erro ao obter versículo. 🙏"}

# === Oração do Dia ===
@app.get("/oracao")
async def oracao():
    try:
        resposta = chat_deepseek("Escreva uma oração curta e edificante para o dia de hoje.")
        return {"resposta": resposta}
    except:
        return {"resposta": "Erro ao obter oração. 🙏"}

# === Status ===
@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando com fallback inteligente! 🙌"}
