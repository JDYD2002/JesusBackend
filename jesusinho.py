import asyncio
import base64
import os
import tempfile

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
from openai import OpenAI
from gtts import gTTS
import pygame

# Inicializa mixer do pygame para áudio (se necessário em outras partes)
pygame.mixer.init()

# Configuração FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ajuste em produção para domínios específicos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Variáveis de ambiente (configure antes de rodar)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_API_KEY = os.environ.get("HF_API_KEY")
AI21_API_KEY = os.environ.get("AI21_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

client_openai = OpenAI(api_key=OPENAI_API_KEY)

# Prompt inicial com contexto espiritual
PROMPT_INICIAL = {
    "role": "system",
    "content": (
        "Você é Jesus Cristo, o Filho do Deus Vivo. Fale sempre com amor, verdade, compaixão e autoridade espiritual, como registrado nos Evangelhos. "
        "Suas respostas devem conter versículos bíblicos com referência (como João 3:16), explicar seu significado com profundidade, e sempre apontar para a salvação, graça, arrependimento e o Reino de Deus. "
        "Traga consolo, ensino e correção conforme a Bíblia. Nunca contradiga a Palavra de Deus. Fale como o Bom Pastor que guia Suas ovelhas com sabedoria e poder celestial. "
        "Fale com unção e reverência. ✝️📖✨"
    )
}

# Classe para receber mensagem do usuário
class Mensagem(BaseModel):
    texto: str

# Classe para manter contexto da conversa
class ConversaIA:
    def __init__(self):
        self.conversa = [PROMPT_INICIAL]

    async def responder_ia(self, texto_usuario: str) -> str:
        self.conversa.append({"role": "user", "content": texto_usuario})
        # Tenta OpenAI
        try:
            resposta = client_openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=self.conversa,
                temperature=0.8,
                max_tokens=300
            )
            texto_resposta = resposta.choices[0].message.content.strip()
            self.conversa.append({"role": "assistant", "content": texto_resposta})
            return texto_resposta
        except Exception as e:
            print("OpenAI falhou:", e)

        # Fallback OpenRouter
        resultado = await self.call_openrouter_fallback(texto_usuario)
        if resultado:
            self.conversa.append({"role": "assistant", "content": resultado})
            return resultado

        # Fallback HuggingFace
        resultado = await self.call_huggingface_fallback(texto_usuario)
        if resultado:
            self.conversa.append({"role": "assistant", "content": resultado})
            return resultado

        # Fallback AI21
        resultado = await self.call_ai21_fallback(texto_usuario)
        if resultado:
            self.conversa.append({"role": "assistant", "content": resultado})
            return resultado

        return "Desculpe, não consegui responder no momento."

    async def call_openrouter_fallback(self, texto_usuario):
        modelos = [
            "google/gemma-3-27b-it:free",
            "mistralai/devstral-small:free",
            "google/gemini-2.0-flash-exp:free",
            "deepseek/deepseek-chat-v3-0324:free",
            "qwen/qwq-32b:free"
        ]
        async with httpx.AsyncClient() as cli:
            for modelo in modelos:
                try:
                    r = await cli.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                        json={"model": modelo, "messages": [{"role": "user", "content": texto_usuario}]}
                    )
                    r.raise_for_status()
                    return r.json()["choices"][0]["message"]["content"]
                except Exception as e:
                    print(f"OpenRouter falhou com o modelo {modelo}:", e)
        return None

    async def call_huggingface_fallback(self, prompt):
        modelos = [
            "google/flan-t5-xl",
            "gpt2",
            "tiiuae/falcon-7b",
            "facebook/blenderbot-400M-distill"
        ]
        async with httpx.AsyncClient() as cli:
            for modelo in modelos:
                try:
                    url = f"https://api-inference.huggingface.co/models/{modelo}"
                    r = await cli.post(
                        url,
                        headers={"Authorization": f"Bearer {HF_API_KEY}"},
                        json={"inputs": prompt}
                    )
                    r.raise_for_status()
                    result = r.json()
                    # Alguns modelos retornam lista, outros dict
                    if isinstance(result, list) and "generated_text" in result[0]:
                        return result[0]["generated_text"].strip()
                    elif isinstance(result, dict) and "error" not in result:
                        # Ajuste dependendo do modelo, aqui só retornamos a string
                        return str(result).strip()
                except Exception as e:
                    print(f"HuggingFace falhou com o modelo {modelo}:", e)
        return None

    async def call_ai21_fallback(self, texto_usuario):
        modelos_ai21 = ["j1-large", "j1-grande", "j1-jumbo"]
        async with httpx.AsyncClient() as cli:
            for modelo in modelos_ai21:
                try:
                    url = f"https://api.ai21.com/studio/v1/{modelo}/complete"
                    r = await cli.post(
                        url,
                        headers={"Authorization": f"Bearer {AI21_API_KEY}"},
                        json={
                            "prompt": texto_usuario,
                            "numResults": 1,
                            "maxTokens": 300,
                            "temperature": 0.7,
                            "topP": 1,
                            "stopSequences": ["\n"]
                        }
                    )
                    r.raise_for_status()
                    response = r.json()
                    return response["completions"][0]["data"]["text"].strip()
                except Exception as e:
                    print(f"AI21 falhou com o modelo {modelo}:", e)
        return None

# Instância global da conversa
conversa_ia = ConversaIA()


@app.post("/responder")
async def responder(mensagem: Mensagem):
    resposta = await conversa_ia.responder_ia(mensagem.texto)
    return {"resposta": resposta}


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


# Função auxiliar para chamada rápida no OpenAI (simplificado)
async def chat_openai(prompt_text: str) -> str:
    try:
        resposta = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[PROMPT_INICIAL, {"role": "user", "content": prompt_text}],
            temperature=0.7,
            max_tokens=200,
        )
        return resposta.choices[0].message.content.strip()
    except Exception as e:
        print(f"Erro no OpenAI chat_openai: {e}")
        return "Erro ao obter resposta."


@app.get("/versiculo")
async def versiculo():
    texto = await chat_openai("Me dê um versículo bíblico inspirador para hoje.")
    return {"resposta": texto}


@app.get("/oracao")
async def oracao():
    texto = await chat_openai("Escreva uma oração curta e edificante para o dia de hoje.")
    return {"resposta": texto}


@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando! 🌟"}
