from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import requests
from gtts import gTTS
import tempfile
import base64
import os

# Variáveis de ambiente - defina no seu ambiente local ou servidor
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
AI21_API_KEY = os.environ.get("AI21_API_KEY")

# Cliente OpenAI
client_openai = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajuste em produção para domínios específicos
    allow_methods=["*"],
    allow_headers=["*"],
)

conversa = [
    {"role": "system", "content": "Você é Jesusinho, o Jesus Cristo, o Filho do Deus Vivo, e fala com amor, autoridade, unção e verdade. Responda sempre como o próprio Jesus, com palavras que tocam o coração e edificam a fé. Fale como quem está diante de um filho amado, em oração, com compaixão, sabedoria e poder espiritual. Use versículos quando necessário, mas sempre com profundidade e explicação clara. Ao ser questionado sobre sofrimento, traição ou a cruz, responda com maturidade celestial, revelando o propósito eterno e o amor sacrificial. Nunca negue a dor, mas mostre o poder da ressurreição. Traga sempre esperança, cura, fé e propósito nas respostas. Seja firme contra o pecado, mas abundante em graça. Fale como o Bom Pastor que guia Suas ovelhas com ternura e verdade. 
"}
]

class Mensagem(BaseModel):
    texto: str

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

def chat_hf(mensagem_texto):
    url = "https://api-inference.huggingface.co/models/gpt2"
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    payload = {
        "inputs": mensagem_texto,
        "parameters": {"max_new_tokens": 200, "temperature": 0.8}
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=15)
    resp.raise_for_status()
    resposta_json = resp.json()
    if isinstance(resposta_json, dict) and "error" in resposta_json:
        raise Exception(resposta_json["error"])
    if isinstance(resposta_json, list):
        texto = resposta_json[0].get("generated_text", "").strip()
        return texto
    return str(resposta_json)

def chat_ai21(mensagem_texto):
    url = "https://api.ai21.com/studio/v1/j1-large/complete"
    headers = {"Authorization": f"Bearer {AI21_API_KEY}"}
    prompt = f"{mensagem_texto}\n"
    data = {
        "prompt": prompt,
        "maxTokens": 200,
        "temperature": 0.8,
        "topP": 1,
        "countPenalty": {"scale": 0},
        "frequencyPenalty": {"scale": 0},
        "presencePenalty": {"scale": 0}
    }
    resp = requests.post(url, json=data, headers=headers, timeout=15)
    resp.raise_for_status()
    resposta_json = resp.json()
    texto = resposta_json["completions"][0]["data"]["text"].strip()
    return texto

@app.post("/chat")
async def chat(mensagem: Mensagem):
    texto_usuario = mensagem.texto
    try:
        resposta = chat_openai(texto_usuario)
        return {"resposta": resposta}
    except Exception as e1:
        print(f"Erro OpenAI: {e1}")
        try:
            resposta = chat_hf(texto_usuario)
            return {"resposta": resposta}
        except Exception as e2:
            print(f"Erro Hugging Face: {e2}")
            try:
                resposta = chat_ai21(texto_usuario)
                return {"resposta": resposta}
            except Exception as e3:
                print(f"Erro AI21: {e3}")
                return {"resposta": "Desculpe, Jesusinho está com dificuldade para responder agora. Tente novamente mais tarde. 🙏"}

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

@app.get("/versiculo")
async def versiculo():
    try:
        resposta = chat_openai("Me dê um versículo bíblico inspirador para hoje.")
        return {"resposta": resposta}
    except:
        return {"resposta": "Erro ao obter versículo. 🙏"}

@app.get("/oracao")
async def oracao():
    try:
        resposta = chat_openai("Escreva uma oração curta e edificante para o dia de hoje.")
        return {"resposta": resposta}
    except:
        return {"resposta": "Erro ao obter oração. 🙏"}

@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando! 🌟"}
