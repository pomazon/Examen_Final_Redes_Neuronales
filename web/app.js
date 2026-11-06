(() => {
  'use strict';

  const DEFAULT_API = 'http://localhost:8000/predict';
  const MAX_BYTES = 10 * 1024 * 1024;

  const ETIQUETAS = {
    ojo_gallo: 'Ojo de gallo',
    roya: 'Roya',
    sana: 'Sana',
  };

  // Métricas del paper (results/resumen.csv) para mostrar info bajo el selector
  const METRICAS = {
    vit_b_16_fondo:  { f1_rocole: 0.587, f1_uganda: 0.439, auc_rocole: 0.920, auc_uganda: 0.777 },
    vit_b_16_basico: { f1_rocole: 0.494, f1_uganda: 0.400, auc_rocole: 0.909, auc_uganda: 0.717 },
    vit_b_16_fuerte: { f1_rocole: 0.460, f1_uganda: 0.408, auc_rocole: 0.920, auc_uganda: 0.750 },
    resnet50_basico: { f1_rocole: 0.696, f1_uganda: 0.662, auc_rocole: 0.831, auc_uganda: 0.743 },
    resnet50_fondo:  { f1_rocole: 0.625, f1_uganda: 0.492, auc_rocole: 0.858, auc_uganda: 0.728 },
    resnet50_fuerte: { f1_rocole: 0.468, f1_uganda: 0.533, auc_rocole: 0.858, auc_uganda: 0.779 },
  };

  const $ = (id) => document.getElementById(id);

  const dropZone      = $('dropZone');
  const fileInput     = $('fileInput');
  const previewWrap   = $('previewContenedor');
  const preview       = $('preview');
  const btnQuitar     = $('btnQuitar');
  const btnPredecir   = $('btnPredecir');
  const btnReintentar = $('btnReintentar');
  const chkGradcam    = $('chkGradcam');
  const chkDemo       = $('chkDemo');
  const inputApi      = $('inputApi');
  const estadoBadge   = $('estadoBadge');
  const selectModelo  = $('selectModelo');
  const infoModelo    = $('infoModelo');

  const estados = {
    vacio:     $('estadoVacio'),
    carga:     $('estadoCarga'),
    error:     $('estadoError'),
    resultado: $('estadoResultado'),
  };

  const clasePredicha   = $('clasePredicha');
  const confianzaTexto  = $('confianzaTexto');
  const barras          = $('barras');
  const binarioWrap     = $('binarioWrap');
  const barraRoyaBin    = $('barraRoyaBin');
  const barraSanaBin    = $('barraSanaBin');
  const valRoyaBin      = $('valRoyaBin');
  const valSanaBin      = $('valSanaBin');
  const gradcamWrap     = $('gradcamContenedor');
  const gradcamImg      = $('gradcamImg');
  const gradcamError    = $('gradcamError');
  const metaResultado   = $('metaResultado');
  const errorMensaje    = $('errorMensaje');

  let archivoActual = null;

  // ---------------- Persistencia ----------------
  const apiGuardada = localStorage.getItem('apiUrl');
  inputApi.value = apiGuardada || DEFAULT_API;
  chkDemo.checked = localStorage.getItem('modoDemo') === '1';
  selectModelo.value = localStorage.getItem('modelo') || 'vit_b_16_fondo';

  inputApi.addEventListener('change', () => localStorage.setItem('apiUrl', inputApi.value.trim()));
  chkDemo.addEventListener('change', () => {
    localStorage.setItem('modoDemo', chkDemo.checked ? '1' : '0');
    actualizarBadge();
  });
  selectModelo.addEventListener('change', () => {
    localStorage.setItem('modelo', selectModelo.value);
    actualizarInfoModelo();
  });

  function actualizarInfoModelo() {
    const m = METRICAS[selectModelo.value];
    if (!m) { infoModelo.textContent = ''; return; }
    infoModelo.textContent =
      `F1 RoCoLe ${m.f1_rocole.toFixed(3)} · F1 Uganda ${m.f1_uganda.toFixed(3)} · ` +
      `AUC RoCoLe ${m.auc_rocole.toFixed(3)} · AUC Uganda ${m.auc_uganda.toFixed(3)}`;
  }

  // ---------------- UI estados ----------------
  function mostrarEstado(nombre) {
    Object.entries(estados).forEach(([k, el]) => { el.hidden = k !== nombre; });
  }
  function actualizarBadge() {
    if (chkDemo.checked) { estadoBadge.textContent = '● Modo demo'; estadoBadge.className = 'badge'; }
    else { estadoBadge.textContent = '● Listo'; estadoBadge.className = 'badge ok'; }
  }
  function setCargando(si) {
    btnPredecir.disabled = si || !archivoActual;
    btnPredecir.textContent = si ? 'Analizando…' : 'Predecir';
  }

  // ---------------- Archivo ----------------
  function validarArchivo(file) {
    if (!file) return 'No se seleccionó ningún archivo.';
    if (!file.type.startsWith('image/')) return 'El archivo debe ser una imagen (JPG o PNG).';
    if (file.size > MAX_BYTES) return 'La imagen supera los 10 MB.';
    return null;
  }
  function cargarArchivo(file) {
    const err = validarArchivo(file);
    if (err) { mostrarError(err); return; }
    archivoActual = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      preview.src = e.target.result;
      previewWrap.hidden = false;
      dropZone.hidden = true;
      btnPredecir.disabled = false;
      mostrarEstado('vacio');
    };
    reader.readAsDataURL(file);
  }
  function limpiarArchivo() {
    archivoActual = null;
    preview.src = '';
    previewWrap.hidden = true;
    dropZone.hidden = false;
    fileInput.value = '';
    btnPredecir.disabled = true;
    mostrarEstado('vacio');
  }

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener('change', (e) => {
    const f = e.target.files?.[0]; if (f) cargarArchivo(f);
  });
  ['dragenter','dragover'].forEach(evt => dropZone.addEventListener(evt, (e) => {
    e.preventDefault(); dropZone.classList.add('arrastrando');
  }));
  ['dragleave','drop'].forEach(evt => dropZone.addEventListener(evt, (e) => {
    e.preventDefault(); dropZone.classList.remove('arrastrando');
  }));
  dropZone.addEventListener('drop', (e) => {
    const f = e.dataTransfer?.files?.[0]; if (f) cargarArchivo(f);
  });

  btnQuitar.addEventListener('click', limpiarArchivo);
  btnReintentar.addEventListener('click', () => { if (archivoActual) predecir(); });

  // ---------------- Predicción ----------------
  btnPredecir.addEventListener('click', predecir);

  async function predecir() {
    if (!archivoActual) return;
    mostrarEstado('carga');
    setCargando(true);
    try {
      const data = chkDemo.checked
        ? await respuestaDemo()
        : await respuestaApi(archivoActual);
      renderResultado(data);
      mostrarEstado('resultado');
      actualizarBadge();
    } catch (e) {
      console.error(e);
      mostrarError(e?.message || 'No se pudo conectar con el backend.');
    } finally {
      setCargando(false);
    }
  }

  async function respuestaApi(file) {
    const url = inputApi.value.trim();
    if (!url) throw new Error('Configura la URL del backend en "Configuración avanzada".');

    const form = new FormData();
    form.append('archivo', file);
    form.append('modelo', selectModelo.value);
    form.append('semilla', '0');
    form.append('gradcam', chkGradcam.checked ? 'true' : 'false');

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 45000);

    let resp;
    try {
      resp = await fetch(url, { method: 'POST', body: form, signal: ctrl.signal });
    } finally {
      clearTimeout(timer);
    }
    if (!resp.ok) {
      const texto = await resp.text().catch(() => '');
      throw new Error(`Backend respondió ${resp.status}. ${texto}`.trim());
    }
    const data = await resp.json();
    if (!data || !data.probabilidades) throw new Error('Respuesta del backend con formato inválido.');
    return data;
  }

  async function respuestaDemo() {
    await new Promise(r => setTimeout(r, 900));
    const clases = ['ojo_gallo', 'roya', 'sana'];
    const probs = clases.map(() => Math.random());
    const s = probs.reduce((a, b) => a + b, 0);
    const norm = probs.map(p => p / s);
    const idx = norm.indexOf(Math.max(...norm));
    const pR = norm[1], pS = norm[2];
    const bin = pR / (pR + pS + 1e-12);
    return {
      prediccion: clases[idx],
      probabilidades: { ojo_gallo: norm[0], roya: norm[1], sana: norm[2] },
      binario: { roya: bin, sana: 1 - bin },
      modelo: selectModelo.value,
      semilla: 0,
      tiempo_ms: Math.round(800 + Math.random() * 300),
      gradcam_base64: chkGradcam.checked ? preview.src : null,
    };
  }

  // ---------------- Render ----------------
  function renderResultado(data) {
    const probs = data.probabilidades;
    const pred = data.prediccion || Object.keys(probs).reduce(
      (a, b) => (probs[a] > probs[b] ? a : b), Object.keys(probs)[0]
    );
    const conf = probs[pred] ?? 0;

    clasePredicha.textContent = ETIQUETAS[pred] || pred;
    confianzaTexto.textContent = `Confianza: ${(conf * 100).toFixed(1)}%`;

    // Barras 3 clases
    const entradas = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    barras.innerHTML = '';
    for (const [nombre, valor] of entradas) {
      const pct = Math.max(0, Math.min(1, valor)) * 100;
      const item = document.createElement('div');
      item.className = 'barraItem' + (nombre === pred ? ' barraItem--top' : '');
      item.innerHTML = `
        <span class="barraItem__nombre">${escapeHtml(ETIQUETAS[nombre] || nombre)}</span>
        <div class="barraItem__pista"><div class="barraItem__relleno" style="width:${pct.toFixed(1)}%"></div></div>
        <span class="barraItem__valor">${pct.toFixed(1)}%</span>`;
      barras.appendChild(item);
    }

    // Vista binaria
    const bin = data.binario || (() => {
      const pR = probs.roya ?? 0, pS = probs.sana ?? 0;
      const v = pR / (pR + pS + 1e-12);
      return { roya: v, sana: 1 - v };
    })();
    binarioWrap.hidden = false;
    barraRoyaBin.style.width = (bin.roya * 100).toFixed(1) + '%';
    barraSanaBin.style.width = (bin.sana * 100).toFixed(1) + '%';
    valRoyaBin.textContent = (bin.roya * 100).toFixed(1) + '%';
    valSanaBin.textContent = (bin.sana * 100).toFixed(1) + '%';

    // Grad-CAM
    if (data.gradcam_base64) {
      gradcamImg.src = data.gradcam_base64;
      gradcamImg.hidden = false;
      gradcamWrap.hidden = false;
      gradcamError.hidden = true;
    } else if (data.gradcam_error) {
      gradcamImg.hidden = true;
      gradcamWrap.hidden = false;
      gradcamError.textContent = 'Grad-CAM no disponible: ' + data.gradcam_error;
      gradcamError.hidden = false;
    } else {
      gradcamWrap.hidden = true;
    }

    const partes = [];
    if (data.modelo) partes.push(`Modelo: <strong>${escapeHtml(data.modelo)}</strong>`);
    if (data.semilla != null) partes.push(`semilla ${data.semilla}`);
    if (data.tiempo_ms != null) partes.push(`inferencia ${data.tiempo_ms} ms`);
    metaResultado.innerHTML = partes.join(' · ');
  }

  function mostrarError(msg) {
    errorMensaje.textContent = msg;
    mostrarEstado('error');
  }
  function escapeHtml(s) {
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // ---------------- Init ----------------
  actualizarBadge();
  actualizarInfoModelo();
  mostrarEstado('vacio');
})();