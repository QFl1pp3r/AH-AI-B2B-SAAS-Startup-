'use strict';

const TOOL_LABELS = {
  separate: 'ВЫДЕЛИТЬ ГОЛОС', denoise: 'ПОЧИСТИТЬ', restore: 'ВОССТАНОВИТЬ РЕЧЬ',
  analyze: 'АНАЛИЗ', inspect: 'ПРОВЕРИТЬ УРОВНИ', transcribe: 'РАСПОЗНАТЬ ТЕКСТ',
};
const ROLE_LABELS = {
  original: 'оригинал', vocals: 'голос', instrumental: 'минус', denoised: 'почищено',
  restored: 'восстановлено', analysis: 'анализ', report: 'уровни', lyrics: 'текст',
};
const $ = (id) => document.getElementById(id);
const view = { selected: null, state: null, side: 'after', peaks: {}, busy: false, agent: null, plans: [], poll: null, root: null, chatError: null, lyrics: null };

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
  renderLyrics(s.lyrics).catch(fail);
  renderTree(s.tree, s.head);
  if (view.root !== s.root.id) { view.root = s.root.id; loadChat(); }
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
    box.append(el('span', 'анализа ещё нет — попроси агента', 'chip ghost'));
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
  if (!selected || selected.kind !== 'json' || ['analysis', 'lyrics'].includes(selected.role)) { box.hidden = true; return; }
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

// The transcript lives only here: it never goes into the agent's context or the chat.
async function renderLyrics(id) {
  const box = $('lyrics');
  if (!id) { box.hidden = true; view.lyrics = null; return; }
  if (!view.lyrics || view.lyrics.id !== id) {
    const info = await api(`/api/artifacts/${id}`);
    view.lyrics = { id, segments: info.content.segments, language: info.content.language };
  }
  const list = $('lyrics-lines');
  list.replaceChildren();
  view.lyrics.segments.forEach((seg) => {
    const item = el('li');
    item.append(el('span', clock(seg.start), 'at'), el('span', seg.text));
    item.addEventListener('click', () => {
      const player = $(`audio-${view.side}`);
      player.currentTime = seg.start;
      highlightLyrics();
    });
    list.append(item);
  });
  if (!view.lyrics.segments.length) list.append(el('li', 'текст не распознан', 'muted'));
  $('lyrics-meta').textContent = `${view.lyrics.language || '?'} · строк: ${view.lyrics.segments.length}`;
  box.hidden = false;
  highlightLyrics();
}

function highlightLyrics() {
  if (!view.lyrics || $('lyrics').hidden) return;
  const t = $(`audio-${view.side}`).currentTime;
  const list = $('lyrics-lines');
  view.lyrics.segments.forEach((seg, i) => {
    const item = list.children[i];
    const now = seg.start <= t && t < seg.end;
    if (now && !item.classList.contains('now')) list.scrollTop = item.offsetTop - list.clientHeight / 3;
    item.classList.toggle('now', now);
  });
}

function renderTree(tree, head) {
  const list = $('tree');
  list.replaceChildren();
  tree.forEach((node) => {
    const item = el('li');
    item.style.marginLeft = `${node.depth * 18}px`;
    item.append(el('span', `${node.depth ? '└ ' : ''}${roleName(node)}`));
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
  const css = getComputedStyle(document.documentElement);
  const [ink, mid, dim] = ['--wave', '--wave-mid', '--wave-dim'].map((name) => css.getPropertyValue(name).trim());
  for (let i = 0; i < count; i++) {
    const peak = data[Math.floor((i / count) * data.length)] || 0;
    const h = Math.max(3, Math.min(1, peak * 1.15) * (height - 6));
    const played = active && i / count < progress;
    ctx.fillStyle = played || (active && !progress) ? ink : (active ? mid : dim);
    roundBar(ctx, i * step, (height - h) / 2, 3, h);
  }
  if (active && progress) {
    ctx.fillStyle = ink;
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
    if (view.side === side) { updatePlay(); drawWave(side); highlightLyrics(); }
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

$('theme').setAttribute('aria-pressed', document.documentElement.classList.contains('emo'));
$('theme').addEventListener('click', () => {
  const emo = document.documentElement.classList.toggle('emo');
  localStorage.setItem('theme', emo ? 'emo' : 'light');
  $('theme').setAttribute('aria-pressed', emo);
  drawBoth();
});

// --- agent chat ------------------------------------------------------------

const STEP_STATE = { pending: 'ждёт', running: 'идёт…', done: 'готово', failed: 'ошибка', skipped: 'пропущен' };

function inputName(ref) {
  if (ref === 'track') return 'трек';
  const [n, role] = ref.split('.');
  return `${ROLE_LABELS[role] || role} из шага ${n}`;
}

async function loadAgent() {
  view.agent = await api('/api/agent');
  const off = $('agent-off');
  $('agent-model').textContent = view.agent.configured ? view.agent.model : '';
  if (view.agent.configured) { off.hidden = true; return; }
  off.replaceChildren(
    el('div', 'Агент выключен: нет ключа провайдера.'),
    el('div', `Задай переменную ${view.agent.key_env} и перезапусти ah ui, или настрой [agent] в ah.toml (OpenAI-совместимый провайдер, Ollama).`),
  );
  off.hidden = false;
}

async function loadChat() {
  if (!view.root) return;
  const chat = await api(`/api/agent/chat?artifact=${encodeURIComponent(view.root)}`);
  const wasRunning = view.plans.filter((p) => p.status === 'running').map((p) => p.id);
  view.plans = chat.plans;
  renderThread();
  const finished = view.plans.find((p) => wasRunning.includes(p.id) && p.status !== 'running');
  if (finished) await load(finished.selected || view.selected);
  const running = view.plans.some((p) => p.status === 'running');
  setBusy(running);
  clearTimeout(view.poll);
  if (running) view.poll = setTimeout(() => loadChat().catch(fail), 1000);
}

function renderThread(pending) {
  const thread = $('thread');
  thread.replaceChildren();
  if (!view.plans.length && !pending && !view.chatError) {
    thread.append(el('p', 'Напиши, что сделать с треком. Агент предложит план, ты его подтвердишь — и он всё сделает сам.', 'empty-thread'));
  }
  view.plans.forEach((plan) => {
    thread.append(el('div', plan.request, 'bubble user'));
    if (plan.reply) thread.append(el('div', plan.reply, 'bubble agent'));
    if (plan.steps.length) thread.append(planCard(plan));
    if (plan.summary) thread.append(el('div', plan.summary, 'bubble agent'));
  });
  if (pending) {
    thread.append(el('div', pending, 'bubble user'));
    thread.append(el('div', 'думаю над планом…', 'bubble agent thinking'));
  } else if (view.chatError) {
    thread.append(el('div', view.chatError.request, 'bubble user'));
    thread.append(el('div', `Не получилось: ${view.chatError.message}`, 'bubble agent error'));
  }
  thread.scrollTop = thread.scrollHeight;
}

function planCard(plan) {
  const card = el('div', undefined, 'plan');
  const list = el('ol');
  plan.steps.forEach((step, i) => {
    const item = el('li', undefined, step.status);
    item.append(el('span', String(i + 1), 'num'));
    const body = el('div');
    body.append(el('div', `${TOOL_LABELS[step.tool] || step.tool.toUpperCase()} ← ${inputName(step.input)}`, 'step-title'));
    if (step.why) body.append(el('div', step.why, 'why'));
    if (step.error) body.append(el('div', step.error, 'err'));
    item.append(body);
    const state = step.status === 'done' && step.seconds != null ? `${STEP_STATE.done} · ${step.seconds} c` : STEP_STATE[step.status];
    item.append(el('span', plan.status === 'proposed' || plan.status === 'cancelled' ? '' : state, 'state'));
    list.append(item);
  });
  card.append(list);
  if (plan.status === 'proposed') {
    const actions = el('div', undefined, 'actions');
    const go = el('button', 'ПОЕХАЛИ →', 'pill dark small');
    const no = el('button', 'ОТМЕНА', 'pill small');
    go.addEventListener('click', () => planAction(plan.id, 'run'));
    no.addEventListener('click', () => planAction(plan.id, 'cancel'));
    actions.append(go, no);
    card.append(actions);
  }
  if (plan.status === 'cancelled') card.append(el('div', 'план отменён', 'final hand'));
  return card;
}

async function planAction(id, action) {
  try {
    await api(`/api/agent/plans/${id}/${action}`, { method: 'POST' });
    view.chatError = null;
  } catch (err) {
    view.chatError = { request: action === 'run' ? 'поехали' : 'отмена', message: err.message };
  }
  await loadChat().catch(fail);
}

$('ask').addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = $('prompt').value.trim();
  if (!text || view.busy) return;
  if (!view.agent || !view.agent.configured) { showStatus('Агент выключен: сначала задай ключ провайдера.', { error: true }); return; }
  setBusy(true);
  view.chatError = null;
  renderThread(text);
  try {
    await postJSON('/api/agent/message', { artifact: view.selected, text });
    $('prompt').value = '';
    hideStatus();
  } catch (err) {
    view.chatError = { request: text, message: err.message };
  } finally {
    setBusy(false);
    await loadChat().catch(fail);
  }
});
$('prompt').addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) $('ask').requestSubmit();
});

loadAgent().catch(fail);
load().catch(fail);
