"""Diapositivas de clase autónomas; el guion del grupo se entrega por separado."""
from pathlib import Path
import base64
from html import escape
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from src.data import ROOT, load_data


def main():
    meta = json.loads((ROOT / "outputs/experiment.json").read_text(encoding="utf-8"))
    results = pd.read_csv(ROOT / "outputs/cv_results.csv")
    audit = json.loads((ROOT / "outputs/data_audit.json").read_text(encoding="utf-8"))
    final_path = ROOT / "outputs/consolidated.json"
    final = json.loads(final_path.read_text(encoding="utf-8")) if final_path.exists() else None
    if final:
        chosen = next(m for m in final['models'] if m['id'] == final['winner_id'])
        meta['best_cv']['f1_macro_mean'] = chosen['f1_macro']
        meta['best_cv']['f1_macro_std'] = chosen['std']
        meta['candidates'] = final['total_configurations']
    train, test = load_data()
    fmt = lambda number: f"{number:.4f}".replace(".", ",")
    best = meta["best_cv"]["f1_macro_mean"]
    grouped = meta["grouped_diagnostic"]["f1_macro_mean"]
    baseline = float(results[results.stage == 3].iloc[0].f1_macro_mean)
    trivial = f1_score(train.target, np.zeros(len(train)), average="macro")
    encoded = base64.b64encode((ROOT / "outputs/figures/pca.png").read_bytes()).decode()
    family_rows = []
    for name, subset in [("k-NN + cocientes", results[results.family == "k-NN"]),
                         ("MLP", results[results.family == "MLP"]),
                         ("SVM RBF", results[results.name.str.startswith("svm_rbf_")]),
                         ("Baseline k-NN", results[results.stage == 3]),
                         ("k-means", results[results.family == "k-means"])]:
        row = subset.iloc[0]
        family_rows.append(f'<div class="bar-row"><span>{escape(name)}</span><div class="track"><i style="width:{row.f1_macro_mean*100:.3f}%"></i></div><strong>{fmt(row.f1_macro_mean)}</strong></div>')
    sections = [
        f'''<section class="slide cover" aria-label="1. Problema">
<p class="kicker">GRUPO 1 · MÉTODOS Y MODELOS MATEMÁTICOS PARA IA</p>
<h1>Once mediciones.<br>Una decisión de calidad.</h1><p class="lead">Clasificación de vinos estándar y superiores<br>con los métodos estudiados durante el semestre.</p>
<div class="metric-line"><div><strong>5.197</strong><span>muestras para aprender</span></div><div><strong>1.300</strong><span>muestras por predecir</span></div><div><strong>11</strong><span>variables originales</span></div></div>
<p class="authors">Juan Alejandro Cardona · Javier Urrego · Santiago Osorio<br>Pontificia Universidad Javeriana Cali · Prof. Tatiana Solano</p></section>''',
        f'''<section class="slide" aria-label="2. Métrica"><p class="kicker">DATOS Y MÉTRICA</p><h2>F1 macro evita ignorar<br>la clase minoritaria.</h2>
<div class="split"><div><div class="class-row"><div style="height:180px;background:var(--blue)"></div><strong>4.369</strong><span>Estándar · 84,1%</span></div><div class="class-row"><div style="height:34px;background:var(--accent)"></div><strong>828</strong><span>Superior · 15,9%</span></div></div>
<div class="rail"><p>Predecir siempre «estándar» daría:</p><strong class="big">84,1%</strong><p>de accuracy, pero solo <b>{fmt(trivial)}</b> de F1 macro.</p><div class="equation">F1 macro = (F1₀ + F1₁) / 2</div><p class="support">Cada clase aporta el mismo peso al resultado.</p></div></div></section>''',
        '''<section class="slide" aria-label="3. Representación"><p class="kicker">REPRESENTACIÓN Y VALIDACIÓN</p><h2>Transformamos las escalas dentro<br>de cada fold de entrenamiento.</h2>
<div class="flow"><div><b>11 mediciones</b><span>id excluido</span></div><div><b>3 cocientes</b><span>hipótesis fisicoquímicas</span></div><div><b>z-score</b><span>escalas comparables</span></div><div><b>Clasificador</b><span>validación sin reajustar</span></div></div>
<div class="split compact"><div><h3>Qué añadimos</h3><p>Acidez fija / volátil<br>SO₂ libre / total<br>Alcohol / densidad</p></div><div><h3>Cómo comparamos</h3><p>Cinco folds estratificados iguales.<br>Semilla 42 y F1 macro.<br>Imputación, escalado y PCA aprenden solo con train del fold.</p></div></div><p class="support">Los valores extremos se describen; no se eliminan sin evidencia de error.</p></section>''',
        f'''<section class="slide" aria-label="4. Métodos"><p class="kicker">PROGRESIÓN DE MODELOS</p><h2>Contrastamos vecinos, centroides,<br>márgenes y redes neuronales.</h2>
<div class="method-list"><div><b>01</b><h3>Baseline</h3><p>k-NN: k = 3, 5, 7<br>Euclidiana / Manhattan<br>k-means: clase por mayoría</p></div><div><b>02</b><h3>Representación</h3><p>Cocientes y centroides<br>Minkowski / coseno<br>PCA: 90% y 95%</p></div><div><b>03</b><h3>Modelos avanzados</h3><p>SVM lineal / poly / RBF<br>MLP ReLU / tanh<br>Regularización y pesos de clase</p></div></div><p class="banner">{meta['candidates']} configuraciones · {meta['folds']} folds comunes · selección por F1 macro</p></section>''',
        f'''<section class="slide" aria-label="5. PCA"><p class="kicker">REDUCCIÓN DIMENSIONAL</p><h2>Conservar varianza no garantiza<br>separar mejor las clases.</h2><img class="pca" src="data:image/png;base64,{encoded}" alt="PCA: proyección de las clases y varianza acumulada por componente">
<p class="support">PCA descriptivo sobre train. Se probaron 90% y 95%; el ganador de la rejilla no usa PCA.</p></section>''',
        f'''<section class="slide" aria-label="6. Resultados"><p class="kicker">RESULTADOS DE VALIDACIÓN</p><h2>k-NN obtuvo el mayor F1 macro<br>entre los candidatos evaluados.</h2><div class="split"><div class="bars">{''.join(family_rows)}<div class="axis"><span>0</span><span>Escala de F1 macro</span><span>1</span></div></div><div class="rail"><strong class="big">{fmt(best)}</strong><p>F1 macro · media de cinco folds<br>Desviación: {fmt(meta['best_cv']['f1_macro_std'])}</p><p><b>11 vecinos · coseno<br>votos por distancia · 3 cocientes</b></p><p class="support">Mejora de {fmt(best-baseline)} frente al mejor baseline.</p></div></div></section>''',
        f'''<section class="slide dark" aria-label="7. Generalización"><p class="kicker">CONTROL DE GENERALIZACIÓN</p><h2>Los perfiles repetidos cambian<br>la dificultad de generalizar.</h2><div class="contrast"><div><strong>{fmt(best)}</strong><span>CV estratificada convencional</span></div><div><strong>{fmt(grouped)}</strong><span>Control con perfiles agrupados</span></div></div>
<p class="lead">{audit['duplicate_feature_rows_train']} filas de train repiten sus once mediciones.</p><p class="support">El control mantiene perfiles idénticos dentro de un mismo grupo. Es un diagnóstico del ganador ya seleccionado, no un test independiente. Seleccionar entre {meta['candidates']} configuraciones puede producir optimismo.</p><p class="banner">El F1 del test oculto todavía no se conoce.</p></section>''',
        '''<section class="slide" aria-label="8. Demostración"><p class="kicker">DEMOSTRACIÓN EN VIVO</p><h2>Un pipeline guardado permite<br>predecir sin entrenar en vivo.</h2><div class="flow three"><div><b>Modelo preparado</b><span>transformaciones + clasificador</span></div><div><b>11 mediciones</b><span>una muestra nueva</span></div><div><b>Predicción</b><span>0 estándar / 1 superior</span></div></div>
<p class="lead">Elegir método · introducir mediciones · predecir calidad.</p><div class="metric-line"><div><strong>App</strong><span>exploración y simulación</span></div><div><strong>CSV</strong><span>1.300 predicciones alineadas</span></div><div><strong>CV</strong><span>evidencia reproducible</span></div></div><p class="support">Los archivos guardan el pipeline completo; no hace falta repetir la búsqueda durante la exposición.</p></section>'''
    ]
    css = '''
:root{--ink:#203747;--accent:#a83d64;--blue:#396d99;--paper:#f8fafb}*{box-sizing:border-box}body{margin:0;background:#152a39;font-family:'Segoe UI',Arial,sans-serif;color:var(--ink)}
.deck{height:calc(100dvh - 60px);display:grid;place-items:center}.slide{display:none;position:relative;width:min(100vw,calc((100dvh - 60px)*16/9));aspect-ratio:16/9;background:var(--paper);padding:4.5% 5%;overflow:hidden;container-type:inline-size}.slide.active{display:flex;flex-direction:column}.kicker{font-size:1.1cqw;font-weight:700;letter-spacing:.14em;color:var(--accent);margin:0 0 1.4cqw}h1,h2{font-size:4.1cqw;line-height:1.12;letter-spacing:-.045em;font-weight:700;margin:0 0 1.8cqw}h1{font-size:5.4cqw}h3{font-size:1.7cqw;margin:.5cqw 0 1.1cqw}p{font-size:1.5cqw;line-height:1.55;margin:.7cqw 0}.lead{font-size:1.9cqw;line-height:1.45}.support{font-size:1.15cqw;color:#5b7080;line-height:1.5}.authors{font-size:1.1cqw;margin-top:auto}.metric-line{display:flex;gap:5cqw;border-top:1px solid #cbd5de;margin-top:2cqw;padding-top:1.5cqw}.metric-line strong{font-size:3cqw;display:block}.metric-line span{font-size:1.1cqw}.split{display:grid;grid-template-columns:1.5fr 1fr;gap:4cqw;align-items:center;flex:1}.split>div:first-child{display:flex;gap:4cqw;align-items:end}.split.compact{grid-template-columns:1fr 1fr;align-items:start}.split.compact>div{display:block}.rail{border-left:3px solid var(--accent);padding-left:2cqw}.big{font-size:5cqw;font-weight:700;line-height:1.2}.equation{font-size:1.8cqw;font-weight:600;padding:1cqw 0;border-top:1px solid #d5dfe5}.class-row{display:flex;flex-direction:column;gap:.4cqw;width:45%;text-align:center}.class-row>div{max-height:14cqw;height:var(--height);width:100%}.class-row strong{font-size:2cqw}.class-row span{font-size:1.3cqw}.flow{display:flex;gap:0;margin:2cqw 0}.flow>div{flex:1;border-top:4px solid var(--blue);padding:1.4cqw 1.4cqw 1cqw 0;margin-right:1.3cqw}.flow b{display:block;font-size:1.8cqw}.flow span{display:block;font-size:1.2cqw;margin-top:.5cqw}.method-list{display:grid;grid-template-columns:repeat(3,1fr);gap:3cqw;margin:1.2cqw 0}.method-list>div{border-top:1px solid #cbd5de;padding-top:1.2cqw}.method-list b{font-size:2cqw;color:var(--accent)}.method-list p{font-size:1.35cqw}.banner{font-size:1.5cqw;border-top:1px solid #cbd5de;padding-top:1.1cqw;margin-top:auto}.pca{width:100%;height:auto;max-height:28cqw;object-fit:contain}.bars{display:block!important}.bar-row{display:grid;grid-template-columns:10.2cqw 1fr 5.2cqw;gap:.8cqw;align-items:center;margin:1.5cqw 0;font-size:1.25cqw}.bar-row strong{text-align:right}.track{height:1.55cqw;background:#e2e9ee}.track i{display:block;height:100%;background:var(--blue)}.bar-row:first-child i{background:var(--accent)}.axis{display:flex;justify-content:space-between;font-size:.95cqw;color:#5b7080;margin-left:11cqw;margin-right:6cqw}.dark{background:#203747;color:white}.dark .support{color:#d0dce6}.dark .kicker{color:#f2a1bc}.contrast{display:flex;gap:6cqw;margin:1.7cqw 0}.contrast strong{display:block;font-size:6.3cqw;letter-spacing:-.05em}.contrast>div:first-child strong{color:#e5eef5}.contrast>div:last-child strong{color:#f3a7c1}.contrast span{font-size:1.4cqw}.dark .banner{border-color:#657b8c}.source{position:absolute;bottom:1.3cqw;right:5%;font-size:.78cqw;color:#738594}.dark .source{color:#c2d0dc}
.controls{height:60px;display:flex;align-items:center;justify-content:center;gap:14px;color:#ecf2f7;background:#102330;font-size:14px}.controls button{background:#304c5f;color:white;border:1px solid #698091;border-radius:5px;font:inherit;padding:8px 13px;cursor:pointer}.controls button:focus-visible{outline:3px solid #f3a7c1}.controls small{color:#bdcfdd}progress{height:6px;width:100px;accent-color:#e482a6}.controls a{color:white}body:fullscreen .deck{height:calc(100dvh - 60px)}
@media print{@page{size:A4 landscape;margin:0}body{background:white}.deck{display:block;height:auto}.slide,.slide.active{display:flex;width:297mm;height:210mm;aspect-ratio:auto;break-after:page;padding:15mm 18mm}.controls{display:none}.source{bottom:7mm}.slide:last-child{break-after:auto}}
'''
    if final:
        sections[2] = sections[2].replace('3 cocientes', '22 atributos').replace('hipótesis fisicoquímicas', '6 derivadas + 5 logaritmos')
        bars = ''.join(f'<div class="bar-row"><span>{escape(m["family"]+" · "+m["id"].replace("selected_", ""))}</span><div class="track"><i style="width:{m["f1_macro"]*100:.3f}%"></i></div><strong>{fmt(m["f1_macro"])}</strong></div>' for m in final['models'])
        sections[5] = f'''<section class="slide" aria-label="6. Resultado consolidado"><p class="kicker">COMPARACIÓN DE TRES PROYECTOS</p><h2>k-NN con centroides obtuvo<br>el mayor F1 macro estándar.</h2><div class="split"><div class="bars">{bars}</div><div class="rail"><strong class="big">{fmt(best)}</strong><p>Media de 5 folds<br>Desviación muestral: {fmt(chosen['std'])}</p><p><b>23 vecinos · coseno<br>votos por distancia · centroides 2/6</b></p><p class="support">22 variables químicas/logarítmicas y 18 coordenadas de distancias. Sin PCA.</p></div></div></section>'''
        sections[6] = f'''<section class="slide dark" aria-label="7. Alcance de la comparación"><p class="kicker">GENERALIZACIÓN</p><h2>Seleccionar con CV no sustituye<br>la evaluación del test oculto.</h2><div class="contrast"><div><strong>116</strong><span>configuraciones existentes revisadas</span></div><div><strong>3</strong><span>proyectos con los mismos datos</span></div></div><p class="lead">Se comparó decisión estándar; los ajustes de umbral se separaron.</p><p class="support">No se repitieron búsquedas largas ni se inventó un ensamble. Hay {audit['duplicate_feature_rows_train']} filas repetidas por perfil. El control agrupado anterior corresponde al modelo base, no al nuevo ganador. Las diferencias de medias no prueban superioridad estadística.</p><p class="banner">F1 del test oculto: pendiente de evaluación docente.</p></section>'''
    # Bars scale with the slide instead of fixed screen pixels.
    sections[1] = sections[1].replace('height:180px', 'height:14cqw').replace('height:34px', 'height:2.65cqw')
    slides = ''.join(section.replace('</section>', '<span class="source">Grupo 1 · guía y resultados de validación del proyecto</span></section>') for section in sections)
    js = '''
const slides=[...document.querySelectorAll('.slide')], counter=document.getElementById('counter'), progress=document.getElementById('progress');let current=0;
function show(index){current=Math.max(0,Math.min(slides.length-1,index));slides.forEach((s,i)=>{s.classList.toggle('active',i===current);s.setAttribute('aria-hidden',i===current?'false':'true')});counter.textContent=`${current+1} / ${slides.length}`;progress.value=current+1;document.getElementById('previous').disabled=current===0;document.getElementById('next').disabled=current===slides.length-1;history.replaceState(null,'',`#${current+1}`)}
async function fullscreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else await document.documentElement.requestFullscreen()}catch{counter.textContent='Usa F11 para pantalla completa'}}
document.getElementById('previous').onclick=()=>show(current-1);document.getElementById('next').onclick=()=>show(current+1);document.getElementById('fullscreen').onclick=fullscreen;document.getElementById('print').onclick=()=>window.print();
document.addEventListener('keydown',e=>{if(['ArrowRight','PageDown',' '].includes(e.key)){e.preventDefault();show(current+1)}else if(['ArrowLeft','PageUp'].includes(e.key)){e.preventDefault();show(current-1)}else if(e.key==='Home')show(0);else if(e.key==='End')show(slides.length-1);else if(e.key.toLowerCase()==='f')fullscreen()});
show((parseInt(location.hash.slice(1),10)||1)-1);
'''
    document = f'''<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Grupo 1 · Calidad del vino</title><style>{css}</style></head><body><main class="deck">{slides}</main><nav class="controls" aria-label="Navegación de diapositivas"><button id="previous">Anterior</button><span id="counter" aria-live="polite"></span><progress id="progress" max="{len(sections)}" value="1"></progress><button id="next">Siguiente</button><button id="fullscreen">Pantalla completa</button><button id="print">Guardar PDF</button><small>Flechas para avanzar · F para ampliar</small></nav><script>{js}</script></body></html>'''
    target = ROOT / "presentacion/PRESENTACION_CLASE.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(document, encoding="utf-8")
    print(f"Presentación independiente: {len(sections)} diapositivas, sin dependencias de internet.")


if __name__ == "__main__":
    main()
