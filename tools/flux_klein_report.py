#!/usr/bin/env python3
"""Compute decoded-image metrics and build an offline FLUX.2 Klein dashboard."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from pathlib import Path

import lpips
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


ORDER = ["bf16", "fp8_scaled", "int8_convrot", "int4_convrot", "nvfp4"]
DISPLAY = {
    "bf16": "BF16 reference",
    "fp8_scaled": "FP8 scaled",
    "int8_convrot": "INT8 ConvRot HQ",
    "int4_convrot": "INT4 ConvRot HQ",
    "nvfp4": "NVFP4 official",
}
SHORT = {
    "bf16": "BF16",
    "fp8_scaled": "FP8 scaled",
    "int8_convrot": "INT8 ConvRot",
    "int4_convrot": "INT4 ConvRot",
    "nvfp4": "NVFP4",
}
COLORS = {
    "bf16": "#94a3b8",
    "fp8_scaled": "#f59e0b",
    "int8_convrot": "#38bdf8",
    "int4_convrot": "#a78bfa",
    "nvfp4": "#34d399",
}


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", required=True)
    parser.add_argument("--latent-report", required=True)
    parser.add_argument("--activation-sensitivity", required=True)
    parser.add_argument("--substitution-sensitivity", required=True)
    parser.add_argument("--routing-sensitivity")
    parser.add_argument("--int4-policy", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-html", required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _image(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


def _tensor(array, device):
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device) * 2.0 - 1.0


def _mean(values):
    return statistics.fmean(values) if values else None


def _relative(path, output_html):
    return Path(os.path.relpath(path, Path(output_html).parent)).as_posix()


def _top_sensitivity(activation_path, substitution_path):
    with open(activation_path, encoding="utf-8") as handle:
        activation = json.load(handle)
    with open(substitution_path, encoding="utf-8") as handle:
        substitution = json.load(handle)
    blocks = [
        {
            "name": name,
            "int4_relative_rmse_pct": values["int4"]["relative_rmse_pct"],
            "int8_relative_rmse_pct": values["int8"]["relative_rmse_pct"],
        }
        for name, values in activation["block_metrics"].items()
    ]
    projections = [
        {
            "name": name,
            "int4_relative_rmse_pct": values["int4"]["relative_rmse_pct"],
            "int8_relative_rmse_pct": values["int8"]["relative_rmse_pct"],
        }
        for name, values in activation["projection_metrics"].items()
    ]
    forward = [
        {
            "name": row["name"],
            "error_energy_increase_pct": row["error_energy_increase_pct"],
            "weight_elements": row["weight_elements"],
        }
        for row in substitution["scopes"]["layers"]
    ]
    return {
        "most_sensitive_blocks": sorted(
            blocks, key=lambda row: row["int4_relative_rmse_pct"], reverse=True
        )[:10],
        "most_sensitive_projections": sorted(
            projections, key=lambda row: row["int4_relative_rmse_pct"], reverse=True
        )[:10],
        "most_damaging_forward_substitutions": sorted(
            forward, key=lambda row: row["error_energy_increase_pct"], reverse=True
        )[:12],
        "least_damaging_forward_substitutions": sorted(
            forward, key=lambda row: row["error_energy_increase_pct"]
        )[:12],
        "activation_settings": activation["settings"],
        "substitution_settings": substitution["settings"],
        "eligible_weights": {row["name"]: row["weight_elements"] for row in substitution["scopes"]["layers"]},
    }


def _routing_sensitivity(path):
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    variant = report["variants"]["int4"]
    rows = variant["scopes"]["blocks"]
    return {
        "method": report["method"],
        "settings": report["settings"],
        "baseline_relative_rmse_pct": variant["baseline"]["relative_rmse_pct"],
        "most_beneficial_blocks": sorted(
            rows, key=lambda row: row["error_energy_reduction_pct"], reverse=True
        )[:12],
        "most_harmful_blocks": sorted(
            rows, key=lambda row: row["error_energy_reduction_pct"]
        )[:8],
    }


def _latent_metrics(path):
    with open(path, encoding="utf-8") as handle:
        report = json.load(handle)
    policy = report["policies"][0]
    return {
        "bf16": {"relative_rmse_pct": 0.0, "cosine": 1.0},
        "fp8_scaled": {
            "relative_rmse_pct": report["controls"]["fp8_scaled"]["relative_rmse_pct"],
            "cosine": report["controls"]["fp8_scaled"]["cosine"],
        },
        "int8_convrot": {
            "relative_rmse_pct": report["controls"]["all_int8"]["relative_rmse_pct"],
            "cosine": report["controls"]["all_int8"]["cosine"],
        },
        "int4_convrot": {
            "relative_rmse_pct": policy["relative_rmse_pct"],
            "cosine": policy["cosine"],
        },
        "nvfp4": {
            "relative_rmse_pct": report["controls"]["nvfp4"]["relative_rmse_pct"],
            "cosine": report["controls"]["nvfp4"]["cosine"],
        },
        "settings": report["settings"],
    }


def _html_template():
    return r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='14' fill='%23071019'/%3E%3Cpath d='M18 12h30v9H28v9h17v9H28v13H18z' fill='%237dd3fc'/%3E%3C/svg%3E">
<title>FLUX.2 Klein 9B · Quantization Lab</title>
<style>
:root{color-scheme:dark;--bg:#07090d;--panel:#0d1119;--panel2:#111722;--line:#222b3a;--text:#edf3fb;--muted:#91a0b5;--cyan:#38bdf8;--violet:#a78bfa;--green:#34d399;--amber:#f59e0b;--shadow:0 24px 70px rgba(0,0,0,.34)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:radial-gradient(1100px 700px at 15% -10%,rgba(56,189,248,.13),transparent 55%),radial-gradient(900px 650px at 92% 2%,rgba(167,139,250,.12),transparent 54%),var(--bg);color:var(--text);font:15px/1.55 Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:#7dd3fc;text-decoration:none}a:hover{text-decoration:underline}.shell{width:min(1500px,calc(100% - 36px));margin:auto}.nav{position:sticky;top:0;z-index:20;backdrop-filter:blur(18px);background:rgba(7,9,13,.78);border-bottom:1px solid rgba(255,255,255,.07)}.nav-inner{height:62px;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{display:flex;align-items:center;gap:11px;font-weight:750;letter-spacing:.01em}.mark{width:30px;height:30px;border-radius:9px;background:linear-gradient(135deg,var(--cyan),var(--violet));box-shadow:0 0 32px rgba(56,189,248,.35);position:relative}.mark:after{content:"";position:absolute;inset:7px;border:2px solid #071019;border-radius:4px}.nav-links{display:flex;gap:18px;font-size:13px;color:var(--muted)}
.hero{padding:78px 0 44px}.eyebrow,.chip{display:inline-flex;align-items:center;gap:8px;border:1px solid rgba(125,211,252,.25);background:rgba(56,189,248,.08);color:#a5e4ff;border-radius:999px;padding:7px 11px;font-size:12px;font-weight:700;letter-spacing:.06em;text-transform:uppercase}.dot{width:7px;height:7px;border-radius:50%;background:var(--green);box-shadow:0 0 14px var(--green)}h1{font-size:clamp(42px,6vw,82px);line-height:.98;letter-spacing:-.055em;max-width:1000px;margin:25px 0 22px}.gradient{background:linear-gradient(92deg,#f8fbff 4%,#7dd3fc 48%,#c4b5fd 86%);-webkit-background-clip:text;background-clip:text;color:transparent}.lead{max-width:840px;color:#aebbd0;font-size:18px}.hero-meta{display:flex;flex-wrap:wrap;gap:9px;margin-top:27px}.chip{border-color:var(--line);background:#0e141e;color:#aebbd0;text-transform:none;letter-spacing:0;font-weight:600}
.section{padding:38px 0}.section-head{display:flex;justify-content:space-between;align-items:end;gap:22px;margin-bottom:20px}.section h2{margin:0;font-size:28px;letter-spacing:-.025em}.section-sub{color:var(--muted);max-width:750px}.grid{display:grid;gap:16px}.kpis{grid-template-columns:repeat(4,1fr)}.card{background:linear-gradient(155deg,rgba(18,24,35,.94),rgba(10,14,21,.94));border:1px solid var(--line);border-radius:18px;box-shadow:var(--shadow)}.kpi{padding:21px;min-height:145px;position:relative;overflow:hidden}.kpi:before{content:"";position:absolute;width:120px;height:120px;border-radius:50%;right:-50px;top:-55px;background:var(--accent,#38bdf8);filter:blur(55px);opacity:.22}.kpi-label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.09em;font-weight:700}.kpi-value{font-size:30px;font-weight:780;letter-spacing:-.04em;margin:9px 0 5px}.kpi-note{font-size:13px;color:#9caac0}.two{grid-template-columns:1.15fr .85fr}.chart-card{padding:22px;min-height:360px}.chart-title{font-weight:720;font-size:17px;margin-bottom:3px}.chart-sub{color:var(--muted);font-size:13px;margin-bottom:18px}.scatter{width:100%;height:280px}.axis{stroke:#354155;stroke-width:1}.gridline{stroke:#1d2634;stroke-width:1}.axis-label{fill:#75859c;font-size:11px}.point-label{fill:#dbe8f7;font-size:11px;font-weight:700}.bars{display:grid;gap:14px}.bar-row{display:grid;grid-template-columns:112px 1fr 70px;align-items:center;gap:10px}.bar-label{color:#c5d2e3;font-size:13px}.bar-track{height:9px;border-radius:99px;background:#1b2432;overflow:hidden}.bar-fill{height:100%;border-radius:inherit;box-shadow:0 0 18px color-mix(in srgb,var(--bar) 32%,transparent);background:var(--bar)}.bar-value{text-align:right;font-variant-numeric:tabular-nums;color:#dce7f5;font-size:13px}.seg{display:flex;gap:6px;background:#0a0e15;border:1px solid var(--line);padding:5px;border-radius:11px}.seg button,.variant-btn{border:0;border-radius:8px;background:transparent;color:#8f9eb2;padding:8px 11px;cursor:pointer;font:inherit;font-size:12px;font-weight:680}.seg button.active,.variant-btn.active{color:#061018;background:#7dd3fc}
.table-wrap{overflow:auto;border-radius:18px;border:1px solid var(--line);background:rgba(12,16,24,.92)}table{border-collapse:collapse;width:100%;min-width:1080px}th,td{padding:15px 14px;border-bottom:1px solid #1d2634;text-align:right;font-variant-numeric:tabular-nums}th{position:sticky;top:0;background:#101722;color:#8493a8;font-size:11px;text-transform:uppercase;letter-spacing:.07em}th:first-child,td:first-child{text-align:left}tr:last-child td{border:0}tbody tr:hover{background:rgba(125,211,252,.035)}.model-cell{display:flex;align-items:center;gap:10px;font-weight:720}.model-swatch{width:9px;height:9px;border-radius:50%;background:var(--swatch);box-shadow:0 0 12px var(--swatch)}.best{color:#6ee7b7}.warn{color:#fbbf24}.muted{color:var(--muted)}
.compare-card{padding:18px}.compare-toolbar{display:flex;align-items:center;justify-content:space-between;gap:15px;flex-wrap:wrap;margin-bottom:14px}.variant-buttons{display:flex;flex-wrap:wrap;gap:7px}.variant-btn{background:#131a25;border:1px solid #263144}.variant-btn.active{background:var(--active-color,#7dd3fc);border-color:transparent}.case-title{font-size:18px;font-weight:750}.case-meta{color:var(--muted);font-size:12px}.compare-stage{position:relative;aspect-ratio:1/1;border-radius:14px;overflow:hidden;background:#05070a;border:1px solid #202a39;isolation:isolate}.compare-stage img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;user-select:none;pointer-events:none}.compare-stage .overlay{clip-path:inset(0 50% 0 0)}.divider{position:absolute;z-index:3;top:0;bottom:0;left:50%;width:2px;background:rgba(255,255,255,.84);box-shadow:0 0 16px rgba(0,0,0,.8)}.divider:after{content:"↔";position:absolute;top:50%;left:50%;translate:-50% -50%;width:34px;height:34px;display:grid;place-items:center;border-radius:50%;background:#f1f5f9;color:#0b1018;font-weight:900;box-shadow:0 5px 18px #000}.slider{position:absolute;z-index:5;inset:0;width:100%;height:100%;opacity:0;cursor:ew-resize;margin:0}.stage-label{position:absolute;z-index:4;top:13px;padding:6px 9px;border-radius:7px;background:rgba(5,8,13,.76);backdrop-filter:blur(9px);font-size:11px;font-weight:800;text-transform:uppercase;letter-spacing:.06em}.stage-label.left{left:13px}.stage-label.right{right:13px}.metrics-strip{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;margin-top:12px}.metric{padding:11px;border-radius:11px;background:#0c121b;border:1px solid #1f2938}.metric-name{font-size:10px;color:#7f8da1;text-transform:uppercase;letter-spacing:.07em}.metric-value{font-size:15px;font-weight:750;margin-top:3px}.thumbs{grid-template-columns:repeat(5,1fr);margin-top:14px}.thumb{display:block;padding:8px;border-radius:12px;background:#0b1018;border:1px solid #1e2836;color:#cbd8e8}.thumb img{display:block;width:100%;aspect-ratio:1;object-fit:cover;border-radius:8px;margin-bottom:7px}.thumb span{font-size:11px;font-weight:700}.prompt{margin-top:13px;color:#93a3b8;font-size:12px;padding:12px 14px;border-left:2px solid #334155;background:#0b1017;border-radius:0 9px 9px 0}
.policy-grid{grid-template-columns:1fr 1fr}.policy-card{padding:22px}.route{display:flex;height:16px;border-radius:999px;overflow:hidden;background:#1e293b;margin:18px 0 10px}.route span{height:100%}.route-legend{display:flex;flex-wrap:wrap;gap:14px;color:#97a6bb;font-size:12px}.route-legend i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:6px}.rank-list{display:grid;gap:8px;margin-top:15px}.rank{display:grid;grid-template-columns:27px 1fr auto;gap:9px;align-items:center;padding:9px 10px;border-radius:9px;background:#0b111a;border:1px solid #1b2533}.rank-num{color:#64748b;font-size:11px}.rank-name{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11px;overflow:hidden;text-overflow:ellipsis}.rank-value{font-size:12px;font-weight:750;color:#fbbf24}.layer-pills{display:flex;flex-wrap:wrap;gap:7px;margin-top:14px}.layer-pill{font:11px ui-monospace,SFMono-Regular,Menlo,monospace;background:#131b28;color:#c4b5fd;border:1px solid #2b3750;border-radius:8px;padding:7px 8px}
.method{grid-template-columns:repeat(3,1fr)}.method-card{padding:20px}.method-card h3{font-size:15px;margin:0 0 8px}.method-card p,.method-card li{color:#94a3b8;font-size:13px}.method-card ul{padding-left:18px;margin-bottom:0}.footer{padding:48px 0 70px;color:#718096;font-size:12px;border-top:1px solid #182130;margin-top:35px}.source-links{display:flex;flex-wrap:wrap;gap:9px;margin-top:15px}.source-links a{padding:7px 9px;border:1px solid #253044;background:#0d131d;border-radius:8px;font-size:11px}
@media(max-width:1050px){.kpis{grid-template-columns:repeat(2,1fr)}.two,.policy-grid{grid-template-columns:1fr}.metrics-strip{grid-template-columns:repeat(3,1fr)}.method{grid-template-columns:1fr}.nav-links{display:none}}@media(max-width:650px){.shell{width:min(100% - 20px,1500px)}.hero{padding-top:52px}.section-head{align-items:stretch;flex-direction:column}.section-head .seg{box-sizing:border-box;display:grid;grid-template-columns:repeat(4,minmax(0,1fr));width:100%}.section-head .seg button{padding:8px 3px}.kpis{grid-template-columns:1fr}.metrics-strip{grid-template-columns:repeat(2,1fr)}.thumbs{grid-template-columns:repeat(2,1fr)}.bar-row{grid-template-columns:88px 1fr 58px}h1{font-size:45px}}
</style>
</head>
<body>
<nav class="nav"><div class="shell nav-inner"><div class="brand"><span class="mark"></span>Quantization Lab</div><div class="nav-links"><a href="#overview">Overview</a><a href="#performance">Performance</a><a href="#compare">Image compare</a><a href="#policy">Layer policy</a><a href="#method">Method</a></div></div></nav>
<main class="shell">
<header class="hero" id="overview"><span class="eyebrow"><span class="dot"></span>Native ComfyUI benchmark complete</span><h1><span class="gradient">FLUX.2 Klein 9B</span><br>precision, measured.</h1><p class="lead">BF16, FP8 scaled, measured INT8 ConvRot, packed mixed INT4 ConvRot, and official NVFP4—compiled or loaded natively, sampled in isolated ComfyUI processes, and compared pixel-for-pixel at four square resolutions.</p><div class="hero-meta" id="heroMeta"></div></header>
<section class="section"><div class="grid kpis" id="kpis"></div></section>
<section class="section" id="performance"><div class="section-head"><div><h2>Performance & fidelity</h2><div class="section-sub">Warm 20-step sampling medians from two timed runs. VRAM is PyTorch peak allocated by the native ComfyUI diffusion pass.</div></div><div class="seg" id="resolutionSeg"></div></div><div class="grid two"><div class="card chart-card"><div class="chart-title">Quality–size frontier</div><div class="chart-sub">Composite closeness uses ½·SSIM + ½·(1−LPIPS), shown as a transparent navigation aid—not a perceptual ground truth.</div><svg class="scatter" id="scatter" viewBox="0 0 650 280" role="img" aria-label="Quality versus model size chart"></svg></div><div class="card chart-card"><div class="chart-title">Selected-resolution runtime</div><div class="chart-sub" id="barSub"></div><div class="bars" id="speedBars"></div><div class="chart-title" style="margin-top:28px">Peak diffusion VRAM</div><div class="bars" id="vramBars" style="margin-top:14px"></div></div></div></section>
<section class="section"><div class="section-head"><div><h2>All numbers, one view</h2><div class="section-sub">Decoded-image metrics are arithmetic means across the four full-resolution comparison cases. Lower LPIPS and latent rRMSE are better; higher SSIM, PSNR, and cosine are better.</div></div></div><div class="table-wrap"><table id="modelTable"></table></div></section>
<section class="section" id="compare"><div class="section-head"><div><h2>Pixel-level comparison cases</h2><div class="section-sub">Drag each canvas to reveal BF16 on the left and the selected quant on the right. Every PNG was generated, decoded, and saved by ComfyUI.</div></div></div><div class="grid" id="compareCases"></div></section>
<section class="section" id="policy"><div class="section-head"><div><h2>Why these layer routes</h2><div class="section-sub">112 transformer projections were profiled on real denoising activations. Complete mixed policies were then tested end-to-end because quantization errors interact across blocks and steps.</div></div></div><div class="grid policy-grid"><div class="card policy-card" id="routingCard"></div><div class="card policy-card" id="sensitivityCard"></div></div></section>
<section class="section" id="method"><div class="section-head"><div><h2>Method & interpretation</h2><div class="section-sub">Reproducible settings and the caveats that keep the comparison honest.</div></div></div><div class="grid method"><article class="card method-card"><h3>Native runtime</h3><ul><li>ComfyUI v<span id="comfyVersion"></span>, its own Python venv</li><li>RTX PRO 6000 Blackwell, CUDA 13</li><li>Euler · flux2 scheduler · CFG 3 · 20 steps</li><li>Shape-specific warmup + two timed passes</li></ul></article><article class="card method-card"><h3>Quality metrics</h3><ul><li>LPIPS AlexNet on full decoded RGB images</li><li>SSIM and PSNR at native output resolution</li><li>512² latent rRMSE on four separate 20-step prompts</li><li>Metrics measure BF16 closeness, not absolute aesthetics</li></ul></article><article class="card method-card"><h3>Kernel truth</h3><p>INT8 uses ComfyUI’s native <code>int8_tensorwise</code> ConvRot layout. INT4 uses true nibble-packed <code>asym_w4a8_int8</code> weights, group-16 Lloyd-Max quantization, FP8 relative scales, and native INT8 activation kernels. Global, modulation, norm, time, and final layers stay BF16.</p></article></div></section>
</main>
<footer class="footer"><div class="shell"><strong>FLUX.2 Klein Quantization Lab</strong><div>Generated locally from measured artifacts. No CDN, remote script, or hidden scoring service is used.</div><div class="source-links"><a href="https://github.com/black-forest-labs/flux2">BFL FLUX.2</a><a href="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B">BF16 model</a><a href="https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9b-nvfp4">Official NVFP4</a><a href="https://github.com/Comfy-Org/ComfyUI/blob/master/comfy/quant_ops.py">ComfyUI quant ops</a><a href="https://github.com/Comfy-Org/comfy-kitchen">comfy-kitchen</a><a href="https://arxiv.org/abs/2512.03673">ConvRot paper</a><a href="https://github.com/feice-huang/ConvRot">ConvRot code</a></div></div></footer>
<script id="reportData" type="application/json">__REPORT_DATA__</script>
<script>
const D=JSON.parse(document.getElementById('reportData').textContent);const order=D.order;const models=D.models;const fmt=(n,d=2)=>n==null?'—':Number(n).toFixed(d);const byRes=(m,r)=>m.cases.find(x=>x.resolution===r);let activeRes=1024;
document.getElementById('heroMeta').innerHTML=[`${D.hardware.gpu}`,`ComfyUI ${D.hardware.comfyui}`,`${D.settings.steps} steps · ${D.settings.scheduler}`,`${D.resolutions.join(' / ')}px`,`5 checkpoints · ${D.total_images} PNGs`].map(x=>`<span class="chip">${x}</span>`).join('');document.getElementById('comfyVersion').textContent=D.hardware.comfyui;
const q=order.slice(1);const bestQ=q.reduce((a,b)=>models[a].quality.closeness>models[b].quality.closeness?a:b);const fp=models.fp8_scaled,iq=models.int8_convrot,w4=models.int4_convrot,nv=models.nvfp4;
const cards=[['Closest quant',models[bestQ].display,`LPIPS ${fmt(models[bestQ].quality.lpips,4)} · SSIM ${fmt(models[bestQ].quality.ssim,4)}`,'#38bdf8'],['Fast quality route',`${fmt(byRes(iq,1024).median_sampling_seconds)} s`,`${fmt(byRes(models.bf16,1024).median_sampling_seconds/byRes(iq,1024).median_sampling_seconds)}× BF16 speed at 1024²`,'#a78bfa'],['Packed INT4',`${fmt(w4.model_bytes/1e9,3)} GB`,`${D.policy.counts.int4} W4 + ${D.policy.counts.int8} W8 + ${D.policy.counts.bf16} BF16 projections · ${fmt(w4.quality.closeness,1)} closeness`,'#a78bfa'],['Smallest / fastest',`${fmt(nv.model_bytes/1e9,3)} GB`,`${fmt(byRes(nv,1024).median_sampling_seconds)} s at 1024² · fidelity tradeoff`,'#34d399']];document.getElementById('kpis').innerHTML=cards.map(c=>`<article class="card kpi" style="--accent:${c[3]}"><div class="kpi-label">${c[0]}</div><div class="kpi-value">${c[1]}</div><div class="kpi-note">${c[2]}</div></article>`).join('');
const seg=document.getElementById('resolutionSeg');D.resolutions.forEach(r=>{const b=document.createElement('button');b.textContent=`${r}px`;b.className=r===activeRes?'active':'';b.onclick=()=>{activeRes=r;[...seg.children].forEach(x=>x.classList.toggle('active',x===b));renderBars()};seg.appendChild(b)});
function renderBars(){const maxS=Math.max(...order.map(k=>byRes(models[k],activeRes).median_sampling_seconds));const maxV=Math.max(...order.map(k=>byRes(models[k],activeRes).diffusion_peak_allocated_gib));document.getElementById('barSub').textContent=`${activeRes}×${activeRes}, median seconds (shorter is better)`;document.getElementById('speedBars').innerHTML=order.map(k=>{const v=byRes(models[k],activeRes).median_sampling_seconds;return `<div class="bar-row"><div class="bar-label">${models[k].short}</div><div class="bar-track"><div class="bar-fill" style="--bar:${models[k].color};width:${v/maxS*100}%"></div></div><div class="bar-value">${fmt(v)} s</div></div>`}).join('');document.getElementById('vramBars').innerHTML=order.map(k=>{const v=byRes(models[k],activeRes).diffusion_peak_allocated_gib;return `<div class="bar-row"><div class="bar-label">${models[k].short}</div><div class="bar-track"><div class="bar-fill" style="--bar:${models[k].color};width:${v/maxV*100}%"></div></div><div class="bar-value">${fmt(v)} GiB</div></div>`}).join('')}renderBars();
function renderScatter(){const svg=document.getElementById('scatter'),W=650,H=280,p={l:52,r:20,t:18,b:38};const xs=order.map(k=>models[k].model_bytes/1e9),ys=order.map(k=>models[k].quality.closeness);const xmin=Math.floor(Math.min(...xs)-1),xmax=Math.ceil(Math.max(...xs)+1),ymin=Math.max(0,Math.floor(Math.min(...ys)/10)*10-5),ymax=102;const X=x=>p.l+(x-xmin)/(xmax-xmin)*(W-p.l-p.r),Y=y=>p.t+(ymax-y)/(ymax-ymin)*(H-p.t-p.b);let s='';for(let i=0;i<5;i++){const y=ymin+(ymax-ymin)*i/4;s+=`<line class="gridline" x1="${p.l}" x2="${W-p.r}" y1="${Y(y)}" y2="${Y(y)}"/><text class="axis-label" x="${p.l-8}" y="${Y(y)+4}" text-anchor="end">${fmt(y,0)}</text>`}for(let i=0;i<5;i++){const x=xmin+(xmax-xmin)*i/4;s+=`<line class="gridline" y1="${p.t}" y2="${H-p.b}" x1="${X(x)}" x2="${X(x)}"/><text class="axis-label" x="${X(x)}" y="${H-15}" text-anchor="middle">${fmt(x,1)} GB</text>`}s+=`<line class="axis" x1="${p.l}" x2="${W-p.r}" y1="${H-p.b}" y2="${H-p.b}"/><line class="axis" x1="${p.l}" x2="${p.l}" y1="${p.t}" y2="${H-p.b}"/>`;order.forEach(k=>{const m=models[k],x=X(m.model_bytes/1e9),y=Y(m.quality.closeness);s+=`<circle cx="${x}" cy="${y}" r="8" fill="${m.color}" stroke="#071019" stroke-width="3"><title>${m.display}: ${fmt(m.quality.closeness,2)}</title></circle><text class="point-label" x="${x+11}" y="${y-10}">${m.short}</text>`});svg.innerHTML=s}renderScatter();
document.getElementById('modelTable').innerHTML=`<thead><tr><th>Checkpoint</th><th>File GB</th><th>Closeness ↑</th><th>LPIPS ↓</th><th>SSIM ↑</th><th>PSNR dB ↑</th><th>Latent rRMSE ↓</th><th>1024 time</th><th>2048 time</th><th>2048 VRAM</th></tr></thead><tbody>${order.map(k=>{const m=models[k];return `<tr><td><div class="model-cell"><span class="model-swatch" style="--swatch:${m.color}"></span>${m.display}</div></td><td>${fmt(m.model_bytes/1e9,3)}</td><td class="${k===bestQ?'best':''}">${fmt(m.quality.closeness,2)}</td><td>${k==='bf16'?'0.0000':fmt(m.quality.lpips,4)}</td><td>${fmt(m.quality.ssim,4)}</td><td>${k==='bf16'?'∞':fmt(m.quality.psnr_db,2)}</td><td>${fmt(m.latent.relative_rmse_pct,2)}%</td><td>${fmt(byRes(m,1024).median_sampling_seconds,2)} s</td><td>${fmt(byRes(m,2048).median_sampling_seconds,2)} s</td><td>${fmt(byRes(m,2048).diffusion_peak_allocated_gib,2)} GiB</td></tr>`}).join('')}</tbody>`;
function metricCells(m,c){return [['Closeness',fmt(c.metrics.closeness,2)],['LPIPS ↓',fmt(c.metrics.lpips,4)],['SSIM ↑',fmt(c.metrics.ssim,4)],['PSNR ↑',`${fmt(c.metrics.psnr_db,2)} dB`],['Sample',`${fmt(c.median_sampling_seconds,2)} s`],['Peak VRAM',`${fmt(c.diffusion_peak_allocated_gib,2)} GiB`]].map(x=>`<div class="metric"><div class="metric-name">${x[0]}</div><div class="metric-value">${x[1]}</div></div>`).join('')}
const cases=document.getElementById('compareCases');D.resolutions.forEach(r=>{const ref=byRes(models.bf16,r);const card=document.createElement('article');card.className='card compare-card';card.dataset.variant='int8_convrot';card.innerHTML=`<div class="compare-toolbar"><div><div class="case-title">${r}×${r} · ${ref.prompt.id.replaceAll('_',' ')}</div><div class="case-meta">Seed ${ref.prompt.seed} · native ${r}px decode</div></div><div class="variant-buttons">${q.map(k=>`<button class="variant-btn ${k==='int8_convrot'?'active':''}" data-v="${k}" style="--active-color:${models[k].color}">${models[k].short}</button>`).join('')}</div></div><div class="compare-stage"><img src="${ref.image}" alt="BF16 ${r}px reference"><img class="overlay" src="${byRes(models.int8_convrot,r).image}" alt="Quantized comparison"><div class="divider"></div><input class="slider" type="range" min="0" max="100" value="50" aria-label="Reveal comparison"><span class="stage-label left">BF16</span><span class="stage-label right">INT8 ConvRot</span></div><div class="metrics-strip">${metricCells(models.int8_convrot,byRes(models.int8_convrot,r))}</div><div class="grid thumbs">${order.map(k=>`<a class="thumb" href="${byRes(models[k],r).image}" target="_blank"><img loading="lazy" src="${byRes(models[k],r).image}" alt="${models[k].display} at ${r}px"><span>${models[k].short}</span></a>`).join('')}</div><div class="prompt">${ref.prompt.text}</div>`;const slider=card.querySelector('.slider'),overlay=card.querySelector('.overlay'),divider=card.querySelector('.divider');slider.oninput=()=>{overlay.style.clipPath=`inset(0 ${100-slider.value}% 0 0)`;divider.style.left=`${slider.value}%`};card.querySelectorAll('.variant-btn').forEach(b=>b.onclick=()=>{const k=b.dataset.v,c=byRes(models[k],r);card.querySelectorAll('.variant-btn').forEach(x=>x.classList.toggle('active',x===b));overlay.src=c.image;card.querySelector('.stage-label.right').textContent=models[k].short;card.querySelector('.metrics-strip').innerHTML=metricCells(models[k],c)});cases.appendChild(card)});
const p=D.policy,s=D.sensitivity,rs=D.routing_sensitivity;const routeParts=[['bf16','#94a3b8','BF16'],['int8',models.int8_convrot.color,'INT8'],['int4',models.int4_convrot.color,'packed INT4']];document.getElementById('routingCard').innerHTML=`<div class="chart-title">Measured routes</div><div class="chart-sub">Global input, modulation, norm, time, and final layers remain BF16 in both presets. Widths below are projection counts; W4 covers ${fmt(p.weight_fractions.int4*100,1)}% of eligible weight elements.</div><div><strong>INT8 HQ</strong><div class="route"><span style="width:100%;background:${models.int8_convrot.color}"></span></div><div class="route-legend"><span><i style="background:${models.int8_convrot.color}"></i>112 INT8 ConvRot projections</span></div></div><div style="margin-top:25px"><strong>Packed INT4 HQ</strong><div class="route">${routeParts.map(x=>`<span style="width:${p.counts[x[0]]/112*100}%;background:${x[1]}"></span>`).join('')}</div><div class="route-legend">${routeParts.map(x=>`<span><i style="background:${x[1]}"></i>${p.counts[x[0]]} ${x[2]}</span>`).join('')}</div><div class="layer-pills">${p.protected_blocks.map(x=>`<span class="layer-pill">${x}</span>`).join('')}</div></div>`;
const ranked=rs?rs.most_beneficial_blocks:s.most_sensitive_blocks;document.getElementById('sensitivityCard').innerHTML=`<div class="chart-title">Measured safeguard impact</div><div class="chart-sub">${rs?`One-block INT8 upgrades from the packed baseline, ${rs.settings.prompts.length} prompts at ${rs.settings.steps} steps. Positive gain reduces final-latent error.`:'Isolated output rRMSE on real activation calls per block.'}</div><div class="rank-list">${ranked.slice(0,8).map((x,i)=>`<div class="rank"><span class="rank-num">${i+1}</span><span class="rank-name">${x.name}</span><span class="rank-value">${rs?(x.error_energy_reduction_pct>=0?'+':'')+fmt(x.error_energy_reduction_pct,2)+'%':fmt(x.int4_relative_rmse_pct,2)+'%'}</span></div>`).join('')}</div>`;
</script>
</body></html>'''


def main():
    args = _args()
    runtime_dir = Path(args.runtime_dir).expanduser().resolve()
    output_json = Path(args.output_json).expanduser().resolve()
    output_html = Path(args.output_html).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_html.parent.mkdir(parents=True, exist_ok=True)
    runtimes = {}
    for key in ORDER:
        with open(runtime_dir / f"{key}.json", encoding="utf-8") as handle:
            runtimes[key] = json.load(handle)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    perceptual = lpips.LPIPS(net="alex", verbose=False).eval().to(device)
    references = {
        row["resolution"]: _image(row["image"])
        for row in runtimes["bf16"]["cases"]
    }
    for key in ORDER:
        for row in runtimes[key]["cases"]:
            ref = references[row["resolution"]]
            if key == "bf16":
                metrics = {
                    "lpips": 0.0,
                    "ssim": 1.0,
                    "psnr_db": None,
                    "pixel_rmse": 0.0,
                    "closeness": 100.0,
                }
            else:
                candidate = _image(row["image"])
                if candidate.shape != ref.shape:
                    raise ValueError(f"Image shape mismatch at {row['resolution']}: {candidate.shape} != {ref.shape}")
                with torch.inference_mode():
                    lpips_value = float(perceptual(_tensor(ref, device), _tensor(candidate, device)).item())
                ssim_value = float(structural_similarity(ref, candidate, data_range=1.0, channel_axis=2))
                psnr_value = float(peak_signal_noise_ratio(ref, candidate, data_range=1.0))
                rmse = float(np.sqrt(np.mean((candidate - ref) ** 2)))
                closeness = 50.0 * (ssim_value + 1.0 - min(max(lpips_value, 0.0), 1.0))
                metrics = {
                    "lpips": lpips_value,
                    "ssim": ssim_value,
                    "psnr_db": psnr_value,
                    "pixel_rmse": rmse,
                    "closeness": closeness,
                }
                print(
                    f"METRIC {key:13} {row['resolution']}px LPIPS={lpips_value:.5f} "
                    f"SSIM={ssim_value:.5f} PSNR={psnr_value:.3f}",
                    flush=True,
                )
            row["metrics"] = metrics
            row["image"] = _relative(row["image"], output_html)
        values = [row["metrics"] for row in runtimes[key]["cases"]]
        runtimes[key]["quality"] = {
            "lpips": _mean([value["lpips"] for value in values]),
            "ssim": _mean([value["ssim"] for value in values]),
            "psnr_db": None if key == "bf16" else _mean([value["psnr_db"] for value in values]),
            "pixel_rmse": _mean([value["pixel_rmse"] for value in values]),
            "closeness": _mean([value["closeness"] for value in values]),
        }
        runtimes[key]["display"] = DISPLAY[key]
        runtimes[key]["short"] = SHORT[key]
        runtimes[key]["color"] = COLORS[key]

    latent = _latent_metrics(args.latent_report)
    for key in ORDER:
        runtimes[key]["latent"] = latent[key]
    with open(args.int4_policy, encoding="utf-8") as handle:
        policy = json.load(handle)
    sensitivity = _top_sensitivity(args.activation_sensitivity, args.substitution_sensitivity)
    weights = sensitivity.pop("eligible_weights")
    policy_default = policy.get("default_mode", "int4")
    routes = {
        name: policy.get("layers", {}).get(name, policy_default)
        for name in weights
    }
    route_counts = {
        mode: sum(route == mode for route in routes.values())
        for mode in ("bf16", "int8", "int4")
    }
    total_weight = sum(weights.values())
    route_weight_fractions = {
        mode: sum(weights[name] for name, route in routes.items() if route == mode) / total_weight
        for mode in ("bf16", "int8", "int4")
    }
    protected_blocks = policy.get("metadata", {}).get("protected_blocks")
    if protected_blocks is None:
        protected_blocks = sorted(policy.get("metadata", {}).get("block_routes", {}))
    first_runtime = runtimes["bf16"]
    report = {
        "schema_version": 1,
        "title": "FLUX.2 Klein 9B Quantization Lab",
        "order": ORDER,
        "resolutions": [row["resolution"] for row in first_runtime["cases"]],
        "total_images": sum(len(runtime["cases"]) for runtime in runtimes.values()),
        "settings": first_runtime["settings"],
        "hardware": {
            "gpu": first_runtime["runtime"]["gpu"],
            "compute_capability": first_runtime["runtime"]["compute_capability"],
            "torch": first_runtime["runtime"]["torch"],
            "cuda": first_runtime["runtime"]["cuda"],
            "comfyui": first_runtime["runtime"]["comfyui"],
            "python_executable": first_runtime["runtime"]["executable"],
            "native_nvfp4_compute": first_runtime["runtime"]["native_nvfp4_compute"],
        },
        "models": runtimes,
        "latent_settings": latent["settings"],
        "policy": {
            "name": policy["name"],
            "counts": route_counts,
            "weight_fractions": route_weight_fractions,
            "int4_layers": sorted(name for name, mode in routes.items() if mode == "int4"),
            "int8_layers": sorted(name for name, mode in routes.items() if mode == "int8"),
            "bf16_layers": sorted(name for name, mode in routes.items() if mode == "bf16"),
            "protected_blocks": protected_blocks,
            "metadata": policy.get("metadata", {}),
        },
        "sensitivity": sensitivity,
        "routing_sensitivity": _routing_sensitivity(args.routing_sensitivity),
        "metric_definitions": {
            "closeness": "100 * (0.5 * SSIM + 0.5 * (1 - clamp(LPIPS, 0, 1)))",
            "decoded_aggregate": "Arithmetic mean across four resolution/prompt cases",
            "latent": "Relative RMSE of final latent versus BF16 on four 512px, 20-step trajectories",
        },
    }
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    compact = json.dumps(report, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    html = _html_template().replace("__REPORT_DATA__", compact)
    with open(output_html, "w", encoding="utf-8") as handle:
        handle.write(html)
    print(f"JSON {output_json}\nHTML {output_html}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
