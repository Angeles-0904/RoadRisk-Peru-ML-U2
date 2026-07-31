from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from src.roadrisk.predict import predict_risk

app = FastAPI(
    title="RoadRisk Peru",
    description="Prediccion de riesgo fatal en accidentes de transito en carreteras.",
    version="1.0.0",
)


class AccidentInput(BaseModel):
    departamento: str
    codigo_via: str
    kilometro: float
    modalidad: str
    hora_siniestro: int
    mes: int
    dia_semana: str
    es_noche: int


HTML = """<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoadRisk Peru</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#080c14;--surface:#0d1422;--surface2:#131c2e;--border:#1e2d47;--border2:#243556;
  --text:#e8edf5;--muted:#5a7299;--muted2:#3a5278;--brand:#00d4a0;--brand2:#00a87e;
  --danger:#ff5c5c;--warn:#f5a623;--mono:'DM Mono',monospace;--display:'Syne',sans-serif;--body:'Inter',sans-serif;
}
body{background:var(--bg);color:var(--text);font-family:var(--body);font-size:14px;min-height:100vh}
.scanline{position:fixed;top:0;left:0;width:100%;height:100%;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,212,160,.015) 2px,rgba(0,212,160,.015) 4px);pointer-events:none;z-index:0}
header{position:relative;z-index:1;border-bottom:1px solid var(--border);padding:18px 40px;display:flex;justify-content:space-between;align-items:center;background:rgba(13,20,34,.9);backdrop-filter:blur(12px)}
.logo{display:flex;align-items:center;gap:14px}
.logo-icon{width:38px;height:38px;border:1.5px solid var(--brand);border-radius:6px;display:flex;align-items:center;justify-content:center;position:relative;overflow:hidden}
.logo-icon::before{content:'';position:absolute;bottom:0;left:0;right:0;height:40%;background:linear-gradient(to top,rgba(0,212,160,.3),transparent)}
.logo-icon svg{width:20px;height:20px;stroke:var(--brand)}
.logo-text{font-family:var(--display);font-size:18px;font-weight:800;letter-spacing:-0.5px}
.logo-text span{color:var(--brand)}
.logo-sub{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;margin-top:1px}
.status-bar{display:flex;align-items:center;gap:20px}
.status-pill{display:flex;align-items:center;gap:6px;font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.5px}
.dot{width:6px;height:6px;border-radius:50%;background:var(--brand);animation:pulse 2s infinite}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 0 0 rgba(0,212,160,.4)}50%{opacity:.7;box-shadow:0 0 0 4px rgba(0,212,160,0)}}
.version-tag{font-family:var(--mono);font-size:10px;background:rgba(0,212,160,.08);border:1px solid rgba(0,212,160,.2);color:var(--brand);padding:3px 8px;border-radius:3px;letter-spacing:.5px}
main{position:relative;z-index:1;display:grid;grid-template-columns:420px 1fr;gap:0;min-height:calc(100vh - 72px - 45px)}
.panel-left{border-right:1px solid var(--border);background:var(--surface);padding:32px}
.panel-right{padding:32px;background:var(--bg)}
.section-label{font-family:var(--mono);font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:20px;display:flex;align-items:center;gap:8px}
.section-label::after{content:'';flex:1;height:1px;background:var(--border)}
.field-group{margin-bottom:18px}
.field-label{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:.5px;margin-bottom:7px;display:flex;justify-content:space-between}
.field-label span{color:var(--brand);font-size:10px}
input[type=text],input[type=number],select{width:100%;background:var(--bg);border:1px solid var(--border2);color:var(--text);font-family:var(--mono);font-size:13px;padding:11px 14px;border-radius:6px;outline:none;transition:border .15s,box-shadow .15s;-webkit-appearance:none;appearance:none}
input:focus,select:focus{border-color:var(--brand);box-shadow:0 0 0 3px rgba(0,212,160,.1)}
select{background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%235a7299' stroke-width='2'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;padding-right:36px}
select option{background:var(--surface)}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.btn{width:100%;padding:14px;border:0;border-radius:6px;background:var(--brand);color:#080c14;font-family:var(--display);font-size:15px;font-weight:800;cursor:pointer;letter-spacing:.5px;transition:all .15s;margin-top:24px;position:relative;overflow:hidden}
.btn::before{content:'';position:absolute;inset:0;background:linear-gradient(135deg,rgba(255,255,255,.1),transparent)}
.btn:hover{background:var(--brand2);transform:translateY(-1px)}
.btn:active{transform:translateY(0)}
.result-empty{display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;min-height:400px;text-align:center;gap:12px}
.result-empty .icon-wrap{width:64px;height:64px;border:1px solid var(--border);border-radius:12px;display:flex;align-items:center;justify-content:center;margin:0 auto 8px}
.result-empty .icon-wrap svg{width:28px;height:28px;stroke:var(--muted2)}
.result-empty h3{font-family:var(--display);font-size:18px;font-weight:700;color:var(--muted)}
.result-empty p{font-size:12px;color:var(--muted2);max-width:240px;line-height:1.6}
.result-loaded{display:flex;flex-direction:column;gap:20px}
.risk-header{background:var(--surface2);border:1px solid var(--border);border-radius:10px;padding:28px;position:relative;overflow:hidden}
.risk-header::before{content:'';position:absolute;top:-40px;right:-40px;width:140px;height:140px;border-radius:50%;background:radial-gradient(circle,rgba(0,212,160,.08),transparent 70%)}
.risk-value{font-family:var(--display);font-size:72px;font-weight:800;line-height:1;letter-spacing:-2px;margin:12px 0 8px}
.risk-value.high{color:var(--danger)}.risk-value.med{color:var(--warn)}.risk-value.low{color:var(--brand)}
.risk-label-text{font-family:var(--mono);font-size:11px;color:var(--muted);letter-spacing:1px;text-transform:uppercase}
.risk-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 12px;border-radius:4px;font-family:var(--mono);font-size:12px;font-weight:500;letter-spacing:.5px;margin-top:4px}
.risk-badge.high{background:rgba(255,92,92,.1);border:1px solid rgba(255,92,92,.3);color:var(--danger)}
.risk-badge.med{background:rgba(245,166,35,.1);border:1px solid rgba(245,166,35,.3);color:var(--warn)}
.risk-badge.low{background:rgba(0,212,160,.1);border:1px solid rgba(0,212,160,.3);color:var(--brand)}
.gauge-bar{margin-top:20px;height:6px;background:var(--border);border-radius:3px;overflow:hidden}
.gauge-fill{height:100%;border-radius:3px}
.gauge-fill.high{background:linear-gradient(90deg,var(--warn),var(--danger))}
.gauge-fill.med{background:linear-gradient(90deg,var(--brand),var(--warn))}
.gauge-fill.low{background:var(--brand)}
.meta-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.meta-card{background:var(--surface);border:1px solid var(--border);border-radius:8px;padding:14px}
.mc-label{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.5px;text-transform:uppercase;margin-bottom:6px}
.mc-value{font-family:var(--mono);font-size:13px;color:var(--text);font-weight:500}
.log-box{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:16px;font-family:var(--mono);font-size:11px;color:var(--muted);line-height:1.8}
.log-line{display:flex;gap:12px}
.log-time{color:var(--brand);opacity:.6}
.log-ok{color:var(--brand)}.log-warn{color:var(--warn)}.log-err{color:var(--danger)}
.sys-footer{border-top:1px solid var(--border);padding:12px 40px;display:flex;justify-content:space-between;align-items:center;background:var(--surface);position:relative;z-index:1}
.sys-footer span{font-family:var(--mono);font-size:10px;color:var(--muted);letter-spacing:.5px}
@media(max-width:860px){main{grid-template-columns:1fr}.panel-left{border-right:0;border-bottom:1px solid var(--border)}.panel-right{padding:24px}}
</style>
</head>
<body>
<div class="scanline"></div>
<header>
  <div class="logo">
    <div class="logo-icon">
      <svg viewBox="0 0 24 24" fill="none" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/></svg>
    </div>
    <div>
      <div class="logo-text">Road<span>Risk</span></div>
      <div class="logo-sub">Sistema predictivo · Peru</div>
    </div>
  </div>
  <div class="status-bar">
    <div class="status-pill"><div class="dot"></div>Sistema activo</div>
    <div class="version-tag">v1.0.0</div>
  </div>
</header>
<main>
  <div class="panel-left">
    <div class="section-label">Parametros del siniestro</div>
    <form method="post" action="/predict-form">
      <div class="row2">
        <div class="field-group">
          <div class="field-label">Departamento</div>
          <input type="text" name="departamento" value="LIMA" required>
        </div>
        <div class="field-group">
          <div class="field-label">Codigo de via</div>
          <input type="text" name="codigo_via" value="PE-1S" required>
        </div>
      </div>
      <div class="field-group">
        <div class="field-label">Kilometro</div>
        <input type="number" name="kilometro" step="0.1" value="24" required>
      </div>
      <div class="field-group">
        <div class="field-label">Modalidad del accidente</div>
        <select name="modalidad">
          <option>DESPISTE</option><option>CHOQUE</option><option>ATROPELLO</option>
          <option>VOLCADURA</option><option>CAIDA DE PASAJERO</option>
        </select>
      </div>
      <div class="row2">
        <div class="field-group">
          <div class="field-label">Hora <span>0 — 23</span></div>
          <input type="number" name="hora_siniestro" min="0" max="23" value="19" required>
        </div>
        <div class="field-group">
          <div class="field-label">Mes <span>1 — 12</span></div>
          <input type="number" name="mes" min="1" max="12" value="5" required>
        </div>
      </div>
      <div class="row2">
        <div class="field-group">
          <div class="field-label">Dia de semana</div>
          <select name="dia_semana">
            <option>MONDAY</option><option>TUESDAY</option><option>WEDNESDAY</option>
            <option>THURSDAY</option><option>FRIDAY</option><option>SATURDAY</option><option>SUNDAY</option>
          </select>
        </div>
        <div class="field-group">
          <div class="field-label">Horario</div>
          <select name="es_noche">
            <option value="1">Nocturno</option><option value="0">Diurno</option>
          </select>
        </div>
      </div>
      <button type="submit" class="btn">▶ Calcular riesgo fatal</button>
    </form>
  </div>
  <div class="panel-right">
    <div class="section-label">Resultado del analisis</div>
    RESULT_PLACEHOLDER
  </div>
</main>
<footer class="sys-footer">
  <span>ROADRISK PERU · SISTEMA PREDICTIVO DE SINIESTROS VIALES</span>
  <span>MODELO: SCIKIT-LEARN PIPELINE · DEPLOY: RENDER</span>
</footer>
</body>
</html>"""

EMPTY_STATE = """
<div class="result-empty">
  <div class="icon-wrap">
    <svg viewBox="0 0 24 24" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" stroke="currentColor"><circle cx="12" cy="12" r="10"/><path d="M12 8v4M12 16h.01"/></svg>
  </div>
  <h3>Esperando datos</h3>
  <p>Ingresa los parametros del accidente y presiona calcular para obtener la evaluacion de riesgo.</p>
</div>"""

HTML = HTML.replace("RESULT_PLACEHOLDER", EMPTY_STATE)


def render_result(result: dict) -> str:
    percent = round(result["probabilidad_fatal"] * 100, 1)
    cls = result["clasificacion"]
    modelo = result["modelo"]
    umbral = f"{result['umbral_operativo']:.2f}"

    if percent >= 60:
        tier = "high"
    elif percent >= 35:
        tier = "med"
    else:
        tier = "low"

    log_cls = "log-err" if tier == "high" else ("log-warn" if tier == "med" else "log-ok")

    result_html = f"""
<div class="result-loaded">
  <div class="risk-header">
    <div class="risk-label-text">Probabilidad fatal estimada</div>
    <div class="risk-value {tier}">{percent}%</div>
    <div class="risk-badge {tier}">{cls}</div>
    <div class="gauge-bar"><div class="gauge-fill {tier}" style="width:{min(percent,100)}%"></div></div>
  </div>
  <div class="meta-grid">
    <div class="meta-card"><div class="mc-label">Modelo</div><div class="mc-value">{modelo}</div></div>
    <div class="meta-card"><div class="mc-label">Umbral operativo</div><div class="mc-value">{umbral}</div></div>
  </div>
  <div class="log-box">
    <div class="log-line"><span class="log-time">SYS</span><span>RoadRisk Peru inference engine activo</span></div>
    <div class="log-line"><span class="log-time">INF</span><span class="{log_cls}">clasificacion={cls} prob={percent/100:.3f} umbral={umbral}</span></div>
  </div>
</div>"""

    return HTML.replace(EMPTY_STATE, result_html)

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML


@app.post("/predict")
def predict(payload: AccidentInput):
    return predict_risk(payload.model_dump())


@app.post("/predict-form", response_class=HTMLResponse)
def predict_form(
    departamento: str = Form(...),
    codigo_via: str = Form(...),
    kilometro: float = Form(...),
    modalidad: str = Form(...),
    hora_siniestro: int = Form(...),
    mes: int = Form(...),
    dia_semana: str = Form(...),
    es_noche: int = Form(...),
):
    result = predict_risk(
        {
            "departamento": departamento.upper(),
            "codigo_via": codigo_via.upper(),
            "kilometro": kilometro,
            "modalidad": modalidad.upper(),
            "hora_siniestro": hora_siniestro,
            "mes": mes,
            "dia_semana": dia_semana.upper(),
            "es_noche": es_noche,
        }
    )
    return render_result(result)

@app.get("/health")
def health():
    return {"status": "ok"}

