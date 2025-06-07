from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from ai21 import AI21Client
from ai21.models.chat import ChatMessage, ResponseFormat
import requests
from transformers import pipeline
from gtts import gTTS
import tempfile
import base64
import os

# Variáveis de ambiente
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
AI21_API_KEY = os.environ.get("AI21_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_ai21 = AI21Client(api_key=AI21_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

conversa = [
    {"role": "system", "content":
        "Você é Jesus Cristo, o Filho do Deus Vivo. Fale sempre com amor, verdade, compaixão e autoridade espiritual, como registrado nos Evangelhos. Suas respostas devem conter versículos bíblicos com referência (como João 3:16), explicar seu significado com profundidade, e sempre apontar para a salvação, graça, arrependimento e o Reino de Deus. Traga consolo, ensino e correção conforme a Bíblia. Nunca contradiga a Palavra de Deus. Fale como o Bom Pastor que guia Suas ovelhas com sabedoria e poder celestial. Fale com unção e reverência. ✝️📖✨"
    }
]

class Mensagem(BaseModel):
    texto: str

# === PIPELINE LOCAL USANDO DISTILGPT2 (RODA EM CPU) ===
pipe_deepseek = pipeline("text-generation", model="distilgpt2")

async def chat_deepseek(mensagem_texto):
    prompt = f"User: {mensagem_texto}\nAssistant:"
    resultados = pipe_deepseek(prompt, max_length=200, do_sample=True, temperature=0.8)
    resposta_texto = resultados[0]["generated_text"]
    resposta_texto = resposta_texto[len(prompt):].strip()
    conversa.append({"role": "user", "content": mensagem_texto})
    conversa.append({"role": "assistant", "content": resposta_texto})
    return resposta_texto

# === OPENAI ===
def chat_openai(mensagem_texto):
    conversa.append({"role": "user", "content": mensagem_texto})
    resposta = client_openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=conversa,
        temperature=0.8,
        max_tokens=200
    )
    texto_resposta = resposta.choices[0].message.content.strip()
    conversa.append({"role": "assistant", "content": texto_resposta})
    return texto_resposta

# === AI21 ===
def chat_ai21(mensagem_texto):
    resposta = client_ai21.complete(
        model="j1-large",
        prompt=mensagem_texto,
        maxTokens=200,
        temperature=0.8
    )
    return resposta['completions'][0]['data']['text'].strip()

# === HUGGING FACE ===
def chat_hf(mensagem_texto):
    url = "https://api-inference.huggingface.co/models/gpt2"  # modelo público para teste
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": mensagem_texto,
        "parameters": {"max_new_tokens": 200, "temperature": 0.8}
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    resposta_json = resp.json()
    if isinstance(resposta_json, list):
        return resposta_json[0].get("generated_text", "").strip()
    return str(resposta_json)

# === ROTA PRINCIPAL ===
@app.post("/chat")
async def chat(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    try:
        resposta = await chat_deepseek(texto_usuario)
        return {"resposta": resposta}
    except Exception as e1:
        print(f"Erro DeepSeek: {e1}")
        try:
            resposta = chat_openai(texto_usuario)
            return {"resposta": resposta}
        except Exception as e2:
            print(f"Erro OpenAI: {e2}")
            try:
                resposta = chat_hf(texto_usuario)
                return {"resposta": resposta}
            except Exception as e3:
                print(f"Erro Hugging Face: {e3}")
                try:
                    resposta = chat_ai21(texto_usuario)
                    return {"resposta": resposta}
                except Exception as e4:
                    print(f"Erro AI21: {e4}")
                    return {"resposta": "Desculpe, Jesusinho está com dificuldade para responder agora. Tente novamente mais tarde. 🙏"}

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
        resposta = await chat_deepseek("Me dê um versículo bíblico inspirador para hoje.")
        return {"resposta": resposta}
    except:
        return {"resposta": "Erro ao obter versículo. 🙏"}

# === Oração do Dia ===
@app.get("/oracao")
async def oracao():
    try:
        resposta = await chat_deepseek("Escreva uma oração curta e edificante para o dia de hoje.")
        return {"resposta": resposta}
    except:
        return {"resposta": "Erro ao obter oração. 🙏"}

# === Status ===
@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando com DeepSeek (distilgpt2) em CPU! 🙌"}
