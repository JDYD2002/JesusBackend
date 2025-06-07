from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import requests
from openai import OpenAI
from gtts import gTTS
import tempfile
import base64
import time

# === Variáveis de ambiente ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI21_API_KEY = os.getenv("AI21_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Mensagem(BaseModel):
    texto: str

# Modelos para fallback em cada API
HF_MODELS = [
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "HuggingFaceH4/zephyr-7b-beta",
]

AI21_MODELS = [
    "j1-large",
    "j1-jumbo",
    "j1-grande",
]

TOGETHER_MODELS = [
    "together-gpt",
    "together-gpt-medium",
    "together-gpt-large",
]

def limpa_resposta(texto, prompt):
    texto_lower = texto.lower().strip()
    prompt_lower = prompt.lower().strip()
    if texto_lower.startswith(prompt_lower):
        return texto[len(prompt):].strip()
    prompt_simples = prompt_lower.replace("responda em português, por favor.\n", "").strip()
    if texto_lower.startswith(prompt_simples):
        return texto[len(prompt_simples):].strip()
    return texto.strip()

def chat_openai(texto, retries=3):
    prompt = f"Responda em português, por favor:\n{texto}"
    for i in range(retries):
        try:
            resp = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}]
            )
            resposta = resp.choices[0].message.content.strip()
            return limpa_resposta(resposta, prompt)
        except Exception as e:
            print(f"Erro no OpenAI gpt-4o-mini, tentativa {i+1}/{retries}: {e}")
            time.sleep(5)
    try:
        resp = client_openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        resposta = resp.choices[0].message.content.strip()
        return limpa_resposta(resposta, prompt)
    except Exception as e:
        print(f"Erro fallback OpenAI gpt-3.5-turbo: {e}")
        return ""

def chat_hf(texto, retries=2):
    prompt = f"Responda em português, por favor:\n{texto}"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    for model in HF_MODELS:
        url = f"https://api-inference.huggingface.co/models/{model}"
        for _ in range(retries):
            try:
                response = requests.post(url, headers=headers, json={"inputs": prompt})
                if response.status_code == 200:
                    data = response.json()
                    # Pode variar, geralmente 'generated_text' ou primeira string no array
                    if isinstance(data, list) and data and "generated_text" in data[0]:
                        resposta = data[0]["generated_text"]
                    else:
                        resposta = data.get("generated_text", "")
                    if resposta:
                        return limpa_resposta(resposta, prompt)
            except Exception as e:
                print(f"Erro HuggingFace {model}: {e}")
            time.sleep(3)
    return ""

def chat_ai21(texto, retries=2):
    prompt = f"Responda em português, por favor:\n{texto}"
    headers = {"Authorization": f"Bearer {AI21_API_KEY}"}
    for model in AI21_MODELS:
        url = f"https://api.ai21.com/studio/v1/{model}/complete"
        payload = {
            "prompt": prompt,
            "maxTokens": 300,
            "temperature": 0.7,
            "topP": 1,
            "stopSequences": ["###"]
        }
        for _ in range(retries):
            try:
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    resposta = data.get("completions", [{}])[0].get("data", {}).get("text", "")
                    if resposta:
                        return limpa_resposta(resposta, prompt)
            except Exception as e:
                print(f"Erro AI21 {model}: {e}")
            time.sleep(3)
    return ""

def chat_together(texto, retries=2):
    prompt = f"Responda em português, por favor:\n{texto}"
    headers = {"Authorization": f"Bearer {TOGETHER_API_KEY}"}
    for model in TOGETHER_MODELS:
        url = f"https://api.together.xyz/llm/{model}"
        payload = {"prompt": prompt, "max_tokens": 300}
        for _ in range(retries):
            try:
                response = requests.post(url, headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    resposta = data.get("choices", [{}])[0].get("text", "")
                    if resposta:
                        return limpa_resposta(resposta, prompt)
            except Exception as e:
                print(f"Erro Together {model}: {e}")
            time.sleep(3)
    return ""

@app.post("/chat")
async def chat_endpoint(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    for func in (chat_openai, chat_hf, chat_ai21, chat_together):
        try:
            resposta = func(texto_usuario)
            if resposta:
                return {"resposta": resposta}
        except Exception as e:
            print(f"Erro {func.__name__}: {e}")
    return {"resposta": "Desculpe, Jesusinho está com dificuldade para responder agora. 🙏"}

@app.post("/tts")
async def tts(mensagem: Mensagem):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            gTTS(text=mensagem.texto, lang="pt-br").save(tmp.name)
            with open(tmp.name, "rb") as f:
                audio_bytes = f.read()
        os.remove(tmp.name)
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        return {"audio_b64": audio_b64}
    except Exception as e:
        return {"audio_b64": None, "erro": str(e)}

@app.get("/versiculo")
async def versiculo():
    prompt = "Me dê um versículo bíblico inspirador e diferente para hoje."
    for func in (chat_openai, chat_hf, chat_ai21, chat_together):
        try:
            resposta = func(prompt)
            if resposta:
                return {"resposta": resposta}
        except Exception as e:
            print(f"Erro versiculo {func.__name__}: {e}")
    return {"resposta": "Erro ao obter versículo. 🙏"}

@app.get("/oracao")
async def oracao():
    prompt = "Oração curta, edificante e diferente para o dia de hoje."
    for func in (chat_openai, chat_hf, chat_ai21, chat_together):
        try:
            resposta = func(prompt)
            if resposta:
                return {"resposta": resposta}
        except Exception as e:
            print(f"Erro oracao {func.__name__}: {e}")
    return {"resposta": "Erro ao obter oração. 🙏"}

@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando com fallback inteligente! 🙌"}
