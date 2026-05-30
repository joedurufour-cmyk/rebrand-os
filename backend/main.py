"""
REBRAND.OS — Motor de Rebranding Determinístico
Pipeline: INGESTA → ANÁLISIS → GAP DETECTION → EVIDENCIA → SCORING → DECISION → OUTPUT → TRACKING
P_entailment ≥ 0.90 | ¬hallucinate | KB1→KB7 architecture
"""
import os, json, re
from datetime import datetime
from typing import Optional, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import openai
import pathlib

app = FastAPI(title="REBRAND.OS", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_PATH = pathlib.Path(__file__).parent.parent / "frontend" / "index.html"

@app.get("/")
async def serve_frontend():
    if FRONTEND_PATH.exists():
        return FileResponse(str(FRONTEND_PATH), media_type="text/html")
    return {"service": "REBRAND.OS API v2", "health": "/api/health"}

KIMI_BASE_URL = "https://api.moonshot.ai/v1"
KIMI_MODEL = "moonshot-v1-128k"

def get_client():
    api_key = os.getenv("KIMI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="KIMI_API_KEY not configured")
    return openai.OpenAI(api_key=api_key, base_url=KIMI_BASE_URL)

@app.get("/api/health")
@app.get("/health")
async def health():
    return {
        "status": "ok", "service": "REBRAND.OS", "version": "2.0.0",
        "kimi_configured": bool(os.getenv("KIMI_API_KEY")),
        "pipeline": "KB1→KB7 | INGESTA→SCORING→OUTPUT"
    }

# ── SCHEMAS ───────────────────────────────────────────────────
class ProfileInput(BaseModel):
    cv_text: str
    job_description: Optional[str] = ""
    target_role: Optional[str] = ""
    target_market: Optional[str] = "remote"
    mode: str = "full_pipeline"  # full_pipeline | score | cv_ats | cv_recruiter | linkedin | freelance | web_research | coach

class ChatMessage(BaseModel):
    cv_text: str
    job_description: Optional[str] = ""
    message: str
    history: Optional[List[dict]] = []

# ── KB SYSTEM PROMPTS ─────────────────────────────────────────
SYSTEM_BASE = """Eres REBRAND.OS — Motor de Rebranding Profesional Determinístico.

PRINCIPIOS ABSOLUTOS (NO NEGOCIABLES):
1. NUNCA inventar experiencia, cargos, empresas, métricas ni certificaciones
2. Si métrica no validada → marcar: ESTIMATED | RANGE | UNKNOWN
3. Separar SIEMPRE: Hecho real | Inferencia | Patch narrativo | Claim no validado
4. P_entailment ≥ 0.90: toda afirmación anclada al CV original
5. Optimizar para: ATS parser + recruiter scan 6-8s + entrevista
6. Mantener coherencia: CV = LinkedIn = Portfolio = Entrevista

ARQUITECTURA KB:
KB1 → ATS rules + mercado actual
KB2 → Posicionamiento narrativo
KB3 → Gap detection + patch narrativo
KB4 → Evidencia técnica validada
KB5 → Freelance + detección scams
KB6 → Scoring + decisión aplicar/no
KB7 → Output generation (NO genera narrativa nueva)

CONTROL DE GENERACIÓN (CRÍTICO):
IF score < 0.50 → NO generar CV completo → sugerir reposicionamiento
IF 0.50-0.74 → versión EXPLORE (con patches marcados)
IF >= 0.75 → versión APPLY optimizada

ERRORES PROHIBIDOS:
- Inventar experiencia
- Inflar métricas sin base
- Generar output sin validación mínima
- Ignorar scoring
- Duplicar narrativa sin control"""

def build_prompt(mode: str, cv: str, jd: str = "", role: str = "", market: str = "remote") -> str:

    if mode == "full_pipeline":
        return f"""EJECUTAR PIPELINE COMPLETO:

CV DEL USUARIO:
{cv}

JOB DESCRIPTION TARGET:
{jd if jd else "No proporcionado"}

ROL TARGET: {role if role else "Inferir del CV"}
MERCADO: {market}

EJECUTAR EN ORDEN:
1. INGESTA: Extraer entidades del CV (roles, empresas, fechas, skills, métricas)
2. ANÁLISIS KB1+KB2: ATS score actual + posicionamiento
3. GAP DETECTION KB3: Brechas vs JD
4. EVIDENCIA KB4: Validar claims
5. SCORING KB6: Calcular fit score 0-100

RESPONDER EN JSON EXACTO:
{{
  "state": "ANÁLISIS COMPLETADO",
  "diagnóstico": {{
    "perfil_actual": "descripción del perfil real extraído",
    "fortalezas": ["fortaleza1 real", "fortaleza2 real"],
    "gaps_criticos": ["gap1 vs JD", "gap2 vs JD"],
    "ats_score_actual": 0,
    "ats_score_proyectado": 0
  }},
  "scoring": {{
    "fit_score": 0,
    "decision": "APPLY | EXPLORE | SKIP",
    "justificacion": "razón basada en CV real",
    "keyword_matches": ["kw1", "kw2"],
    "keyword_gaps": ["gap1", "gap2"]
  }},
  "patches": [
    {{"seccion": "SUMMARY", "original": "texto original", "patch": "versión optimizada", "tipo": "Hecho real | Inferencia | Patch narrativo"}},
    {{"seccion": "EXPERIENCE", "original": "texto original", "patch": "versión ATS", "tipo": "Hecho real"}}
  ],
  "next_action": "acción recomendada específica"
}}"""

    elif mode == "cv_ats":
        return f"""GENERAR CV VERSIÓN ATS:

CV ORIGINAL:
{cv}

JD TARGET:
{jd if jd else "Optimizar para rol: " + role}

REGLAS ATS:
- Headers estándar: PROFESSIONAL SUMMARY | EXPERIENCE | SKILLS | EDUCATION
- Sin tablas, columnas, headers/footers, gráficos
- Keywords del JD integradas naturalmente
- Action verbs + cuantificación SOLO donde el CV original lo soporte
- Marcar con [ESTIMATED] métricas inferidas
- Marcar con [SUGGESTED] skills no confirmados en CV

FORMATO: Texto plano ATS-safe, máximo 2 páginas equivalente.
RESPONDER: Solo el CV completo sin explicaciones adicionales."""

    elif mode == "cv_recruiter":
        return f"""GENERAR CV VERSIÓN RECRUITER (VISUAL):

CV ORIGINAL:
{cv}

JD TARGET:
{jd if jd else "Optimizar para rol: " + role}

REGLAS RECRUITER (6-8 segundos de scan):
- Summary de impacto en 3 líneas MAX
- Bullets de logros con formato: Verbo + Qué + Resultado cuantificado (si existe en CV)
- Skills relevantes al JD primero
- Diseño mental: jerarquía visual clara
- Primera sección debe capturar atención inmediata

RESPONDER: CV completo optimizado para recruiter humano."""

    elif mode == "linkedin":
        return f"""OPTIMIZAR PERFIL LINKEDIN:

CV ORIGINAL:
{cv}

ROL TARGET: {role if role else "Inferir del CV"}
MERCADO: {market}

GENERAR:
1. HEADLINE (120 chars max): keyword-rich, valor claro
2. ABOUT (2000 chars max): narrativa en primera persona, hook en primera línea
3. EXPERIENCE: bullets de cada rol optimizados para LinkedIn search
4. SKILLS TOP 10: ordenadas por relevancia ATS + mercado actual
5. KEYWORDS ADICIONALES: para aparecer en búsquedas de recruiters

Coherencia total con CV. ¬inventar."""

    elif mode == "freelance":
        return f"""ANÁLISIS FREELANCE + DETECCIÓN DE OPORTUNIDADES:

CV ORIGINAL:
{cv}

MERCADO TARGET: {market}

ANALIZAR Y RESPONDER:
1. PLATAFORMAS RECOMENDADAS: Upwork/Fiverr/Toptal/LinkedIn/otras según skills reales del CV
2. SERVICIOS OFRECIBLES: basados estrictamente en experiencia confirmada en CV
3. RANGO DE PRECIOS: estimado por mercado actual (marcar como ESTIMATED)
4. PROPUESTA DE VALOR: diferenciador único basado en CV real
5. RED FLAGS SCAM: patrones a evitar en plataformas
6. PROPUESTA TIPO: template de propuesta para cliente basado en skills reales
7. GAPS PARA FREELANCE: qué falta para competir en mercado actual

JSON RESPONSE:
{{
  "plataformas": [{{"nombre": "", "fit": "HIGH|MED|LOW", "razon": ""}}],
  "servicios": [{{"servicio": "", "base": "Hecho real|Inferencia", "precio_estimado": ""}}],
  "propuesta_valor": "",
  "red_flags": ["flag1", "flag2"],
  "propuesta_template": "",
  "gaps_freelance": ["gap1", "gap2"]
}}"""

    elif mode == "score":
        return f"""SCORING RÁPIDO CV vs JD:

CV:
{cv}

JD:
{jd}

CALCULAR:
- fit_score: 0-100
- keyword_matches: lista
- keyword_gaps: lista críticos
- decision: APPLY (>=75) | EXPLORE (50-74) | SKIP (<50)
- razon: 2 líneas máximo

JSON SOLO."""

    return f"Analiza este CV y JD. CV: {cv[:500]} JD: {jd[:300]}"


# ── MAIN PIPELINE ENDPOINT ────────────────────────────────────
@app.post("/api/v1/pipeline")
async def run_pipeline(data: ProfileInput):
    if not data.cv_text.strip():
        raise HTTPException(status_code=400, detail="cv_text requerido")

    client = get_client()
    prompt = build_prompt(data.mode, data.cv_text, data.job_description, data.target_role, data.target_market)

    try:
        response = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_BASE},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=6000,
        )
        raw = response.choices[0].message.content

        # Parse JSON if expected
        parsed = None
        if data.mode in ["full_pipeline", "freelance", "score"]:
            try:
                clean = re.sub(r'```(?:json)?\n?', '', raw).strip().rstrip('`')
                parsed = json.loads(clean)
            except:
                parsed = {"raw": raw}

        return {
            "status": "PASS",
            "mode": data.mode,
            "data": {"content": raw, "structured": parsed},
            "meta": {"model": KIMI_MODEL, "tokens": response.usage.total_tokens},
            "transducers": [
                {"action": "download", "label": "Descargar TXT", "endpoint": "/api/v1/export"},
                {"action": "copy", "label": "Copiar"}
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error LLM: {str(e)}")


# ── COACH CHAT ────────────────────────────────────────────────
@app.post("/api/v1/coach")
async def coach(data: ChatMessage):
    if not data.message.strip():
        raise HTTPException(status_code=400, detail="message requerido")

    client = get_client()
    messages = [{"role": "system", "content": SYSTEM_BASE}]

    if data.cv_text:
        messages.append({"role": "user", "content": f"CV en contexto:\n{data.cv_text}"})
        messages.append({"role": "assistant", "content": "CV cargado. ¿Cómo puedo ayudarte?"})

    if data.job_description:
        messages.append({"role": "user", "content": f"JD target:\n{data.job_description}"})
        messages.append({"role": "assistant", "content": "JD cargado."})

    for h in (data.history or [])[-6:]:
        if h.get("role") in ["user", "assistant"]:
            messages.append(h)

    messages.append({"role": "user", "content": data.message})

    try:
        response = client.chat.completions.create(
            model=KIMI_MODEL,
            messages=messages,
            temperature=0.4,
            max_tokens=3000,
        )
        return {
            "status": "PASS",
            "reply": response.choices[0].message.content,
            "tokens": response.usage.total_tokens
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Error: {str(e)}")


# ── EXPORT ────────────────────────────────────────────────────
@app.post("/api/v1/export")
async def export_txt(body: dict):
    content = body.get("content", "")
    name = body.get("name", "cv_rebrand_os")
    txt = content.encode("utf-8")
    return StreamingResponse(
        iter([txt]),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{name}_{datetime.utcnow().strftime("%Y%m%d")}.txt"'}
    )
