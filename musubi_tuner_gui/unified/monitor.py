"""Self-polling live training monitor (dashboard + last 100 log lines + download log).

No ``gr.Timer``: in Gradio 6 every event, including a timer tick, runs the frontend's per-component
status bookkeeping, which on a page this size costs hundreds of milliseconds of browser time per
tick. Instead one ``gr.HTML`` component polls its own server function once per second while it is
on screen and patches its DOM directly. When nothing changed the reply is a few bytes: the dashboard
(a few KB) is sent when the run's state changed, the log tail only when new lines were committed
(a progress-bar redraw sends just that one line), and the elapsed clock ticks in the browser.
"""

from __future__ import annotations

import html
import time

import gradio as gr

from .dashboard import render_dashboard
from .downloads import DOWNLOADS
from .runner import RUNNER

LOG_LINES = 100


def monitor_poll(seen):
    """Server function called from the browser: ``seen`` = [run version, download version, log version]."""
    try:
        seen_version, seen_download, seen_log = (list(seen) + [-1, -1, -1])[:3] if isinstance(seen, (list, tuple)) else (-1, -1, -1)
    except Exception:
        seen_version, seen_download, seen_log = -1, -1, -1
    snapshot = RUNNER.snapshot()
    download_version, download_status, download_text = DOWNLOADS.snapshot()
    active = bool(snapshot["active"])
    reply = {"v": snapshot["version"], "lv": snapshot["log_version"], "dv": download_version, "active": active, "now": time.time()}
    if snapshot["version"] != seen_version:
        reply["dash"] = render_dashboard(snapshot)
        reply["pl"] = snapshot["progress_line"]
    if snapshot["log_version"] != seen_log:
        reply["log"] = RUNNER.log_text(LOG_LINES)
    if download_version != seen_download:
        reply["dl"] = download_text if download_status != "idle" else ""
    return reply


def build_monitor(root_id: str) -> gr.HTML:
    snapshot = RUNNER.snapshot()
    initial = (
        '<div class="ud-monitor">'
        f'<div class="mon-dash">{render_dashboard(snapshot)}</div>'
        '<div class="mon-log-wrap">'
        '<div class="mon-log-head"><span>📟 Training Status · last 100 log lines</span>'
        '<button type="button" class="mon-copy" title="Copy the log">📋 Copy</button></div>'
        f'<pre class="mon-log" tabindex="0">{html.escape(RUNNER.log_tail(LOG_LINES))}</pre>'
        "</div></div>"
    )
    return gr.HTML(
        initial,
        js_on_load=MONITOR_JS.replace("__ROOT__", root_id),
        server_functions=[monitor_poll],
        elem_classes=["ws-monitor"],
    )


MONITOR_JS = r"""
const root = document.getElementById('__ROOT__');
const dash = element.querySelector('.mon-dash');
const log = element.querySelector('.mon-log');
const copy = element.querySelector('.mon-copy');
let version = -1, logVersion = -1, downloadVersion = -1, inflight = false, inflightAt = 0, request = 0, ticks = 0, offset = 0;
let lines = '', progress = '';  // committed log lines + the live progress-bar line, shown together
// The download log box sits in the Model Files section, which is not on the page while collapsed:
// keep the latest text and write it whenever the box is there.
let downloadText = null;
const showDownload = () => {
  const box = downloadText !== null && root ? root.querySelector('.dl-log') : null;
  if (!box || box.textContent === downloadText) return;
  box.textContent = downloadText;
  box.scrollTop = box.scrollHeight;
  box.classList.toggle('has-text', !!downloadText);
};

// Elapsed time and "… ago" texts tick here, so an unchanged run costs the server nothing to re-render.
const pad = (n) => String(n).padStart(2, '0');
const duration = (s) => {
  s = Math.max(0, Math.floor(s));
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600), m = Math.floor((s % 3600) / 60), r = s % 60;
  return d ? `${d}d ${pad(h)}:${pad(m)}:${pad(r)}` : `${h}:${pad(m)}:${pad(r)}`;
};
const ago = (s) => (s < 60 ? `${Math.floor(s)} s ago` : s < 3600 ? `${Math.floor(s / 60)} min ago` : `${(s / 3600).toFixed(1)} h ago`);
const clock = () => {
  const now = Date.now() / 1000 + offset;
  dash.querySelectorAll('[data-since]').forEach((el) => { const t = duration(now - Number(el.dataset.since)); if (el.textContent !== t) el.textContent = t; });
  dash.querySelectorAll('[data-ago]').forEach((el) => { const t = ago(Math.max(0, now - Number(el.dataset.ago))); if (el.textContent !== t) el.textContent = t; });
};

// Minimal DOM patch: keeps unchanged nodes (so CSS transitions such as the progress bar animate).
const morph = (target, html) => {
  const next = document.createElement('div');
  next.innerHTML = html;
  const patch = (a, b) => {
    if (a.nodeType !== b.nodeType || a.nodeName !== b.nodeName) { a.replaceWith(b.cloneNode(true)); return; }
    if (a.nodeType === 3) { if (a.nodeValue !== b.nodeValue) a.nodeValue = b.nodeValue; return; }
    if (a.nodeType !== 1) return;
    for (const attr of Array.from(a.attributes)) if (!b.hasAttribute(attr.name)) a.removeAttribute(attr.name);
    for (const attr of Array.from(b.attributes)) if (a.getAttribute(attr.name) !== attr.value) a.setAttribute(attr.name, attr.value);
    const ac = Array.from(a.childNodes), bc = Array.from(b.childNodes);
    for (let i = 0; i < bc.length; i++) { if (i < ac.length) patch(ac[i], bc[i]); else a.appendChild(bc[i].cloneNode(true)); }
    for (let i = ac.length - 1; i >= bc.length; i--) ac[i].remove();
  };
  const ac = Array.from(target.childNodes), bc = Array.from(next.childNodes);
  if (ac.length !== bc.length) { target.innerHTML = html; return; }
  for (let i = 0; i < bc.length; i++) patch(ac[i], bc[i]);
};

const tick = async (force) => {
  if (inflight && Date.now() - inflightAt < 20000) return;  // a request the network lost must not stop the monitor
  if (!force && (document.visibilityState !== 'visible' || element.offsetParent === null)) return;
  inflight = true;
  inflightAt = Date.now();
  const id = ++request;
  try {
    // every 15 s the dashboard is re-rendered anyway (stall warning, server-side "… ago" texts)
    const refresh = force || ++ticks % 15 === 0;
    const reply = await server.monitor_poll([refresh ? -1 : version, downloadVersion, logVersion]);
    if (reply && id === request) {  // a late reply of an abandoned request is older than the current state
      if (typeof reply.now === 'number') offset = reply.now - Date.now() / 1000;
      if (reply.dash !== undefined) { morph(dash, reply.dash); clock(); }
      if (reply.log !== undefined || reply.pl !== undefined) {
        if (reply.log !== undefined) { lines = reply.log; logVersion = reply.lv; }
        if (reply.pl !== undefined) progress = reply.pl;
        const text = progress ? (lines ? lines + '\n' + progress : progress) : lines;
        if (log.textContent !== text) {
          const stick = log.scrollTop + log.clientHeight >= log.scrollHeight - 12;
          log.textContent = text;
          if (stick) log.scrollTop = log.scrollHeight;
        }
      }
      version = reply.v;
      if (root) {
        root.setAttribute('data-running', reply.active ? '1' : '0');
        // a stop question left open when the run ended must not greet the next run
        if (!reply.active && root.getAttribute('data-stopask') === '1') root.setAttribute('data-stopask', '0');
      }
      if (reply.dl !== undefined) { downloadVersion = reply.dv; downloadText = reply.dl; }
      showDownload();
    }
  } catch (error) {
    // the server restarted or the page is closing: try again on the next tick
  } finally {
    if (id === request) inflight = false;
  }
};

if (copy) copy.addEventListener('click', async () => {
  const text = log.textContent || '';
  let done = false;
  try { if (navigator.clipboard && window.isSecureContext) { await navigator.clipboard.writeText(text); done = true; } } catch (e) {}
  if (!done) {  // plain http from another PC (LAN address): no clipboard API there, the selection copy still works
    const area = document.createElement('textarea');
    area.value = text;
    area.setAttribute('readonly', '');
    area.style.position = 'fixed';
    area.style.opacity = '0';
    document.body.appendChild(area);
    area.select();
    try { done = document.execCommand('copy'); } catch (e) {}
    area.remove();
  }
  copy.textContent = done ? '✅ Copied' : '⚠️ Copy failed';
  setTimeout(() => (copy.textContent = '📋 Copy'), 1500);
});
log.scrollTop = log.scrollHeight;
window.__udPoll = window.__udPoll || {};
window.__udPoll['__ROOT__'] = () => { tick(true); setTimeout(() => tick(true), 700); };
tick(true);
setInterval(() => tick(false), 1000);
setInterval(() => { if (document.visibilityState === 'visible' && element.offsetParent !== null) clock(); }, 1000);
"""
