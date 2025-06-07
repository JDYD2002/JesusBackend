from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from ai21 import AI21Client
from together import Together
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

client_openai = OpenAI(api_key=OPENAI_API_KEY)
client_ai21 = AI21Client(api_key=AI21_API_KEY)
together_client = Together(api_key=TOGETHER_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Histórico da conversa ===
conversa = [
    {"role": "system", "content":
        "Você é Jesus Cristo, o Filho do Deus Vivo. Fale sempre com amor, verdade, compaixão e autoridade espiritual, como registrado nos Evangelhos. Suas respostas devem conter versículos bíblicos com referência (como João 3:16), explicar seu significado com profundidade, e sempre apontar para a salvação, graça, arrependimento e o Reino de Deus. Traga consolo, ensino e correção conforme a Bíblia. Nunca contradiga a Palavra de Deus. Fale como o Bom Pastor que guia Suas ovelhas com sabedoria e poder celestial. Fale com unção e reverência. ✝️📖✨"
    }
]

class Mensagem(BaseModel):
    texto: str

# === DeepSeek ===
def chat_deepseek(texto):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": conversa + [{"role": "user", "content": texto}],
        "temperature": 0.8,
        "max_tokens": 200
    }
    response = requests.post(url, json=payload, headers=headers)
    response.raise_for_status()
    resposta_texto = response.json()["choices"][0]["message"]["content"]
    conversa.append({"role": "user", "content": texto})
    conversa.append({"role": "assistant", "content": resposta_texto})
    return resposta_texto.strip()

# === OpenAI ===
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

# === Hugging Face ===
def chat_hf(mensagem_texto):
    url = "https://api-inference.huggingface.co/models/gpt2"
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

# === Together AI ===
def chat_together(mensagem_texto):
    resp = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=conversa + [{"role": "user", "content": mensagem_texto}],
        stream=False,
    )
    resposta = resp.choices[0].message.content.strip()
    conversa.append({"role": "user", "content": mensagem_texto})
    conversa.append({"role": "assistant", "content": resposta})
    return resposta

# === ROTA PRINCIPAL ===
@app.post("/chat")
async def chat(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    try:
        resposta = chat_deepseek(texto_usuario)
        return {"resposta": resposta, "modelo": "DeepSeek"}
    except Exception as e1:
        print(f"Erro DeepSeek: {e1}")
        try:
            resposta = chat_openai(texto_usuario)
            return {"resposta": resposta, "modelo": "OpenAI"}
        except Exception as e2:
            print(f"Erro OpenAI: {e2}")
            try:
                resposta = chat_hf(texto_usuario)
                return {"resposta": resposta, "modelo": "HuggingFace"}
            except Exception as e3:
                print(f"Erro Hugging Face: {e3}")
                try:
                    resposta = chat_together(texto_usuario)
                    return {"resposta": resposta, "modelo": "Together AI"}
                except Exception as e4:
                    print(f"Erro Together: {e4}")
                    try:
                        resposta = chat_ai21(texto_usuario)
                        return {"resposta": resposta, "modelo": "AI21"}
                    except Exception as e5:
                        print(f"Erro AI21: {e5}")
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
