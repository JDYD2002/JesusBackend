import os
import base64
import tempfile
import asyncio
from gtts import gTTS
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import httpx

# 🚀 NEW TOOLS
from dotenv import load_dotenv
from loguru import logger
from apscheduler.schedulers.background import BackgroundScheduler
from prometheus_fastapi_instrumentator import Instrumentator
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

# 🔧 Load env vars
load_dotenv()

# 🔐 API KEYS
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")
AI21_API_KEY = os.getenv("AI21_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# 🤖 OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ⚙️ Logging setup
logger.add("jesusinho.log", rotation="1 MB")
logger.info("🚀 Iniciando o Jesusinho API...")

# 📊 DB Setup
Base = declarative_base()
engine = create_engine('sqlite:///conversas.db')
Session = sessionmaker(bind=engine)
session = Session()

class Conversa(Base):
    __tablename__ = 'conversas'
    id = Column(Integer, primary_key=True)
    entrada = Column(String)
    resposta = Column(String)
    data = Column(DateTime, default=datetime.now)

Base.metadata.create_all(engine)

# 📅 Agendador (exemplo: loga um versículo todo dia 8h)
scheduler = BackgroundScheduler()

def versiculo_diario():
    logger.info("📖 Versículo do dia disparado.")

scheduler.add_job(versiculo_diario, 'cron', hour=8)
scheduler.start()

# 🔥 FastAPI setup
app = FastAPI()

# 📊 Prometheus monitoring
Instrumentator().instrument(app).expose(app)

# 🌍 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📖 Início da conversa
conversa = [{
    "role": "system",
    "content": (
        "Você é Jesus Cristo, o Filho do Deus Vivo. Responda sempre em português brasileiro. "
        "Fale com amor, verdade, compaixão e autoridade espiritual, como registrado nos Evangelhos. "
        "Use versículos bíblicos e explique com profundidade, trazendo consolo, ensino e correção. ✝️📖✨"
    )
}]

# 📩 Modelo de entrada
class Mensagem(BaseModel):
    texto: str

    async def responder_ia(self, texto_usuario: str):
        # Adiciona input do usuário na conversa
        conversa.append({"role": "user", "content": texto_usuario})

        # --- 1️⃣ OpenAI ---
        try:
            resposta = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=conversa,
                temperature=0.8,
                max_tokens=300
            )
            texto_resposta = resposta.choices[0].message.content.strip()
            conversa.append({"role": "assistant", "content": texto_resposta})

            # Salvar no banco de dados
            with Session() as db_session:
                nova = Conversa(entrada=texto_usuario, resposta=texto_resposta)
                db_session.add(nova)
                db_session.commit()

            logger.info(f"🧠 Resposta OpenAI: {texto_resposta}")
            return texto_resposta
        except Exception as e:
            logger.error(f"❌ OpenAI falhou: {e}")

        # --- 2️⃣ Fallback OpenRouter ---
        async def call_openrouter():
            modelos = [
                "mistralai/devstral-small:free",
                "google/gemini-2.0-flash-exp:free",
                "google/gemma-3-27b-it:free",
                "microsoft/mai-ds-r1:free",
                "qwen/qwen3-14b:free",
                "mistralai/mistral-nemo:free",
                "meta-llama/llama-4-maverick:free",
                "qwen/qwen3-32b:free",
                "nvidia/llama-3.1-nemotron-ultra-253b-v1:free",
                "qwen/qwen-2.5-72b-instruct:free",
            ]
            async with httpx.AsyncClient(timeout=15) as cli:
                for modelo in modelos:
                    try:
                        r = await cli.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}"},
                            json={
                                "model": modelo,
                                "messages": [
                                    {"role": "system", "content": "Você é Jesus Cristo, IA em português."},
                                    {"role": "user", "content": texto_usuario}
                                ]
                            }
                        )
                        r.raise_for_status()
                        resp_json = r.json()
                        resposta = resp_json["choices"][0]["message"]["content"].strip()
                        conversa.append({"role": "assistant", "content": resposta})
                        return resposta
                    except httpx.HTTPStatusError as e:
                        logger.warning(f"⚠️ OpenRouter {modelo} falhou: {e}")
                    await asyncio.sleep(1)
            return None

        # --- 3️⃣ Fallback HuggingFace ---
        async def call_huggingface():
            prompt = f"Jesus Cristo (IA): {texto_usuario}"
            modelos = [
                "HuggingFaceH4/zephyr-7b-beta",
                "meta-llama/Llama-3.2-1B-Instruct",
                "microsoft/Phi-3.5-mini-instruct",
                "unsloth/Llama-3.2-1B-Instruct"
            ]
            async with httpx.AsyncClient(timeout=20) as cli:
                for modelo in modelos:
                    try:
                        r = await cli.post(
                            f"https://api-inference.huggingface.co/models/{modelo}",
                            headers={"Authorization": f"Bearer {HF_API_KEY}"},
                            json={"inputs": prompt}
                        )
                        r.raise_for_status()
                        result = r.json()
                        if isinstance(result, list) and "generated_text" in result[0]:
                            resposta = result[0]["generated_text"].strip()
                            conversa.append({"role": "assistant", "content": resposta})
                            return resposta
                    except httpx.HTTPStatusError as e:
                        logger.warning(f"⚠️ HuggingFace {modelo} falhou: {e}")
                    await asyncio.sleep(1)
            return None

        # --- 4️⃣ Fallback AI21 ---
        async def call_ai21():
            prompt = f"Jesus: {texto_usuario}"
            modelo = "j1-jumbo"
            async with httpx.AsyncClient(timeout=20) as cli:
                try:
                    r = await cli.post(
                        f"https://api.ai21.com/studio/v1/{modelo}/complete",
                        headers={"Authorization": f"Bearer {AI21_API_KEY}"},
                        json={
                            "prompt": prompt,
                            "numResults": 1,
                            "maxTokens": 300,
                            "temperature": 0.7,
                            "stopSequences": ["\n"]
                        }
                    )
                    r.raise_for_status()
                    resposta = r.json()["completions"][0]["data"]["text"].strip()
                    conversa.append({"role": "assistant", "content": resposta})
                    return resposta
                except httpx.HTTPStatusError as e:
                    logger.warning(f"⚠️ AI21 falhou: {e}")
            return None

        # --- Executa fallbacks em ordem ---
        for func in [call_openrouter, call_huggingface, call_ai21]:
            resultado = await func()
            if resultado:
                return resultado

        # --- Nenhum serviço respondeu ---
        return "Desculpe, não consegui responder no momento. 🙏"

# ROTAS ----------------------------------------------------------------------

@app.post("/responder")
async def responder(mensagem: Mensagem):
    resposta = await mensagem.responder_ia(mensagem.texto)
    return {"resposta": resposta}

@app.post("/tts")
async def tts(mensagem: Mensagem):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmpfile:
            gtts = gTTS(text=mensagem.texto, lang="pt-br")
            gtts.save(tmpfile.name)
            with open(tmpfile.name, "rb") as f:
                audio_bytes = f.read()
            os.remove(tmpfile.name)
        return {"audio_b64": base64.b64encode(audio_bytes).decode("utf-8")}
    except Exception as e:
        logger.error(f"Erro no TTS: {e}")
        return {"audio_b64": None, "erro": str(e)}

@app.get("/versiculo")
async def versiculo():
    try:
        return {"resposta": conversa[-1]["content"]}
    except:
        return {"resposta": "Erro ao obter versículo. 🙏"}

@app.get("/oracao")
async def oracao():
    return {"resposta": "Senhor Deus, abençoa este dia com Tua graça e misericórdia. Amém."}

@app.get("/")
async def raiz():
    return {"mensagem": "API Jesusinho está rodando! 🌟"}




