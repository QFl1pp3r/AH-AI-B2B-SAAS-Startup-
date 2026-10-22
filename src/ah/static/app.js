'use strict';

const TOOL_LABELS = {
  separate: ['ВЫДЕЛИТЬ ГОЛОС', 'выделяю голос'],
  denoise: ['ПОЧИСТИТЬ', 'чищу запись'],
  restore: ['ВОССТАНОВИТЬ РЕЧЬ', 'восстанавливаю речь'],
  analyze: ['АНАЛИЗ', 'слушаю темп и аккорды'],
  inspect: ['ПРОВЕРИТЬ УРОВНИ', 'меряю уровни'],
};
const ROLE_LABELS = {
  original: 'оригинал', vocals: 'голос', instrumental: 'минус', denoised: 'почищено',
  restored: 'восстановлено', analysis: 'анализ', report: 'уровни',
};
const $ = (id) => document.getElementById(id);
const view = { selected: null, state: null, side: 'after', peaks: {}, busy: false };

async function api(path, options = {}) {
  const init = { ...options, headers: { ...(options.headers || {}) } };
  if (init.method === 'POST') init.headers['X-AH-UI'] = '1';
  const response = await fetch(path, init);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || body.error || `HTTP ${response.status}`);
  return body;
}

function postJSON(path, payload) {
  return api(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
}

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function clock(seconds) {
  const s = Math.max(0, Math.floor(seconds || 0));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
}

function roleName(artifact) {
  return ROLE_LABELS[artifact.role] || artifact.role;
}

function showStatus(text, { error = false, warnings = [] } = {}) {
  const box = $('status');
  box.replaceChildren(el('div', text));
  if (warnings.length) {
    const list = el('ul');
    warnings.forEach((w) => list.append(el('li', w)));
    box.append(list);
  }
  box.classList.toggle('error', error);
  box.hidden = false;
}

function hideStatus() { $('status').hidden = true; }

async function load(selected) {
  view.state = await api(`/api/state${selected ? `?selected=${encodeURIComponent(selected)}` : ''}`);
  view.selected = view.state.selected;
  render();
}

function render() {
  const s = view.state;
  $('empty').hidden = Boolean(s.head);
  $('work').hidden = !s.head;
  $('export').hidden = !s.head;
  if (!s.head) return;

  const audio = s.audio;
  $('title').textContent = s.root.label || s.root.id;
  $('meta').textContent = `${clock(audio.meta.duration_s)} · ${roleName(audio)}`;
  $('after-label').textContent = audio.tool === 'import' ? 'оригинал' : `${roleName(audio)} (${audio.tool})`;
  $('before-block').hidden = !s.before;
  if (s.before) $('before-label').textContent = roleName(s.before);
  $('export').href = `/api/artifacts/${audio.id}/audio?download=1`;

  setSource('after', audio.id);
  setSource('before', s.before ? s.before.id : null);
  if (!s.before) setSide('after');
  drawBoth();
  renderAnalysis(s.analysis);
  renderReport();
  renderTools(s.tools);
  renderTree(s.tree, s.head);
}

function setSource(side, id) {
  const player = $(`audio-${side}`);
  const src = id ? `/api/artifacts/${id}/audio` : '';
  if (player.dataset.id === (id || '')) return;
  player.pause();
  player.dataset.id = id || '';
  if (src) player.src = src; else player.removeAttribute('src');
  if (id && !view.peaks[id]) {
    api(`/api/artifacts/${id}/peaks?bins=240`).then((r) => { view.peaks[id] = r.peaks; drawBoth(); });
  }
}

function renderAnalysis(analysis) {
  const box = $('analysis');
  box.replaceChildren();
  if (!analysis) {
    box.append(el('span', 'анализа ещё нет — нажми «анализ»', 'chip ghost'));
    return;
  }
  if (analysis.bpm) box.append(el('span', `${Math.round(analysis.bpm)} BPM`, 'chip'));
  if (analysis.meter) box.append(el('span', analysis.meter, 'chip'));
  if (analysis.key) box.append(el('span', analysis.key, 'chip'));
  if (analysis.chords.length) box.append(el('span', analysis.chords.slice(0, 8).join(' – '), 'chip'));
  if (analysis.of !== view.state.audio.id) box.append(el('span', 'анализ версии-предка', 'chip ghost'));
}

const LEVEL_LABELS = {
  duration_s: ['длительность', (v) => clock(v)],
  peak_dbfs: ['пик', (v) => `${v} dBFS`],
  rms_dbfs: ['RMS', (v) => `${v} dBFS`],
  clipped_ratio: ['клиппинг', (v) => `${Math.round(v * 1000) / 10}%`],
  silence_ratio: ['тишина', (v) => `${Math.round(v * 100)}%`],
};

async function renderReport() {
  const box = $('report');
  const selected = view.state.tree.find((a) => a.id === view.selected);
  // Analysis already shows as chips; only level reports and unknown JSON get a panel.
  if (!selected || selected.kind !== 'json' || selected.role === 'analysis') { box.hidden = true; return; }
  const info = await api(`/api/artifacts/${selected.id}`);
  box.replaceChildren();
  const known = Object.keys(LEVEL_LABELS).filter((key) => key in info.content);
  if (known.length) {
    known.forEach((key) => {
      const [name, format] = LEVEL_LABELS[key];
      box.append(el('span', `${name}: ${info.content[key] === null ? '—' : format(info.content[key])}`, 'chip'));
    });
    box.className = 'chips report';
  } else {
    box.textContent = JSON.stringify(info.content, null, 2);
    box.className = 'report raw';
  }
  box.hidden = false;
}

function renderTools(tools) {
  const box = $('tools');
  if (box.childElementCount === tools.length) return;
  box.replaceChildren();
  tools.forEach((tool) => {
    const [label] = TOOL_LABELS[tool.name] || [tool.name.toUpperCase()];
    const button = el('button', '', 'tool');
    button.append(el('span', label), el('span', '→'));
    button.title = tool.summary;
    button.addEventListener('click', () => runTool(tool.name));
    box.append(button);
  });
}

function renderTree(tree, head) {
  const list = $('tree');
  list.replaceChildren();
  tree.forEach((node) => {
    const item = el('li');
    item.style.marginLeft = `${node.depth * 18}px`;
    item.append(el('span', `${node.depth ? '└ ' : ''}${roleName(node)}${node.kind === 'json' ? '' : ''}`));
    item.append(el('span', node.id, 'id'));
    if (node.id === head) { item.classList.add('head'); item.append(el('span', ' ← текущая')); }
    if (node.id === view.selected) item.classList.add('selected');
    item.addEventListener('click', () => load(node.id).catch(fail));
    list.append(item);
  });
}

function drawBoth() {
  drawWave('after');
  if (view.state && view.state.before) drawWave('before');
}

function drawWave(side) {
  const canvas = $(`wave-${side}`);
  const player = $(`audio-${side}`);
  const data = view.peaks[player.dataset.id];
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  if (!width) return;
  canvas.width = width * ratio;
  canvas.height = height * ratio;
  const ctx = canvas.getContext('2d');
  ctx.scale(ratio, ratio);
  ctx.clearRect(0, 0, width, height);
  if (!data) return;
  const step = 5;
  const count = Math.floor(width / step);
  const progress = player.duration ? player.currentTime / player.duration : 0;
  const active = view.side === side;
  for (let i = 0; i < count; i++) {
    const peak = data[Math.floor((i / count) * data.length)] || 0;
    const h = Math.max(3, Math.min(1, peak * 1.15) * (height - 6));
    const played = active && i / count < progress;
    ctx.fillStyle = played || (active && !progress) ? '#111' : (active ? '#4a4a46' : '#b9b9b4');
    roundBar(ctx, i * step, (height - h) / 2, 3, h);
  }
  if (active && progress) {
    ctx.fillStyle = '#111';
    ctx.fillRect(progress * width - 1, 0, 2, height);
  }
}

function roundBar(ctx, x, y, w, h) {
  ctx.beginPath();
  ctx.roundRect ? ctx.roundRect(x, y, w, h, 1.5) : ctx.rect(x, y, w, h);
  ctx.fill();
}

function setSide(side) {
  const from = $(`audio-${view.side}`);
  const to = $(`audio-${side}`);
  const playing = !from.paused;
  if (side !== view.side) {
    from.pause();
    if (from.currentTime && to.duration) to.currentTime = Math.min(from.currentTime, to.duration);
  }
  view.side = side;
  $('ab-after').classList.toggle('on', side === 'after');
  $('ab-before').classList.toggle('on', side === 'before');
  if (playing) to.play();
  updatePlay();
  drawBoth();
}

function updatePlay() {
  const player = $(`audio-${view.side}`);
  $('play').textContent = player.paused ? '▶' : '❚❚';
  $('clock').textContent = `${clock(player.currentTime)} / ${clock(player.duration)}`;
}

async function runTool(name) {
  if (view.busy) return;
  let params;
  try {
    params = JSON.parse($('params').value || '{}');
  } catch {
    showStatus('Параметры — не JSON. Поправь поле «параметры инструмента».', { error: true });
    return;
  }
  const [, doing] = TOOL_LABELS[name] || [name, name];
  const started = Date.now();
  setBusy(true);
  const timer = setInterval(() => showStatus(`${doing}… ${clock((Date.now() - started) / 1000)}`), 500);
  showStatus(`${doing}…`);
  try {
    const result = await postJSON(`/api/run/${name}`, { artifact: view.selected, params });
    clearInterval(timer);
    const warnings = result.run.warnings || [];
    showStatus(`готово за ${clock((Date.now() - started) / 1000)}`, { warnings });
    if (!warnings.length) setTimeout(hideStatus, 2500);
    await load(result.selected);
  } catch (err) {
    clearInterval(timer);
    fail(err);
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  view.busy = busy;
  document.body.classList.toggle('busy', busy);
}

function fail(err) {
  showStatus(err.message, { error: true });
}

async function upload(file) {
  if (!file || view.busy) return;
  const form = new FormData();
  form.append('file', file);
  setBusy(true);
  showStatus(`загружаю ${file.name}…`);
  try {
    const artifact = await api('/api/import', { method: 'POST', body: form });
    hideStatus();
    await load(artifact.id);
  } catch (err) {
    fail(err);
  } finally {
    setBusy(false);
  }
}

$('file').addEventListener('change', (e) => { upload(e.target.files[0]); e.target.value = ''; });
['dragenter', 'dragover'].forEach((type) => window.addEventListener(type, (e) => {
  e.preventDefault(); document.body.classList.add('dragging');
}));
['dragleave', 'drop'].forEach((type) => window.addEventListener(type, (e) => {
  e.preventDefault(); document.body.classList.remove('dragging');
}));
window.addEventListener('drop', (e) => upload(e.dataTransfer.files[0]));

$('play').addEventListener('click', () => {
  const player = $(`audio-${view.side}`);
  if (player.paused) player.play(); else player.pause();
});
$('ab-after').addEventListener('click', () => setSide('after'));
$('ab-before').addEventListener('click', () => setSide('before'));
['after', 'before'].forEach((side) => {
  const player = $(`audio-${side}`);
  ['timeupdate', 'play', 'pause', 'loadedmetadata', 'ended'].forEach((type) => player.addEventListener(type, () => {
    if (view.side === side) { updatePlay(); drawWave(side); }
  }));
  $(`wave-${side}`).addEventListener('click', (e) => {
    if (view.side !== side) setSide(side);
    if (!player.duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    player.currentTime = ((e.clientX - rect.left) / rect.width) * player.duration;
  });
});

$('rollback').addEventListener('click', async () => {
  try { const r = await postJSON('/api/rollback', {}); await load(r.head); } catch (err) { fail(err); }
});
$('make-head').addEventListener('click', async () => {
  try { const r = await postJSON('/api/head', { artifact: view.selected }); await load(r.head); } catch (err) { fail(err); }
});
window.addEventListener('resize', drawBoth);

load().catch(fail);
