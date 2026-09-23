from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime

from jinja2 import BaseLoader, Environment, select_autoescape

from model_radar.analysis import _benchmark_family_key, model_type
from model_radar.models import ModelRecord, Snapshot, View

_STYLE = (
    ":root{color-scheme:dark;--ink:#e6edf3;--muted:#9aa9b8;--line:#33404d;"
    "--paper:#18222d;--canvas:#0d141b;--navy:#d9e7f2;--teal:#66d5c5;--mint:#193c3c;"
    "--hover:#21303d;--control:#111b24;--surface-muted:#111b24;--row-line:#293744;"
    "--link-bg:#12302f;--link-border:#326f6b;--on-accent:#071116;--open-row:#1a2930;--open-row-hover:#20343b;"
    "--amber:#f4c56d;--amber-bg:#3c311c;--rose:#ff9aa9;--rose-bg:#3b2028;"
    "--shadow:0 12px 34px rgba(0,0,0,.28)}"
    "body[data-theme='light']{color-scheme:light;--ink:#18212f;--muted:#667085;--line:#e4e7ec;"
    "--paper:#fff;--canvas:#f4f6f8;--navy:#17324d;--teal:#087f8c;--mint:#dff4ef;"
    "--hover:#f3f7f8;--control:#fff;--surface-muted:#f8fbfc;--row-line:#eef0f3;"
    "--link-bg:#f0fbf8;--link-border:#b7e1db;--on-accent:#fff;--open-row:#eff8f6;--open-row-hover:#e5f2ef;"
    "--amber:#a15c00;--amber-bg:#fff4db;--rose:#a33d52;--rose-bg:#fff0f2;"
    "--shadow:0 12px 34px rgba(25,45,65,.08)}"
    "*{box-sizing:border-box}body{margin:0;background:var(--canvas);color:var(--ink);"
    "font-family:'Avenir Next','Segoe UI',sans-serif;line-height:1.5}"
    ".page{max-width:1280px;margin:0 auto;padding:32px 24px 56px}"
    ".report-meta{display:flex;align-items:center;justify-content:space-between;gap:12px;margin:0 0 18px;color:var(--muted);font-size:.8rem;font-weight:700}"
    ".theme-toggle{display:inline-grid;place-items:center;width:34px;height:34px;border:1px solid var(--line);border-radius:9px;background:var(--paper);color:var(--ink);font-size:1.05rem;cursor:pointer}"
    ".theme-toggle:hover{border-color:var(--teal);color:var(--teal)}"
    ".highlights{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;margin:0}"
    ".highlight{display:flex;flex-direction:column;gap:2px;padding:16px 18px;border:1px solid var(--line);"
    "border-radius:16px;background:var(--paper);box-shadow:var(--shadow);color:inherit;font:inherit;text-align:left;cursor:pointer}"
    ".highlight:hover{border-color:var(--teal)}.highlight:focus-visible{outline:2px solid var(--teal);outline-offset:2px}"
    ".highlight-label{color:var(--muted);font-size:.66rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}"
    ".highlight-model{margin-top:4px;color:var(--ink);font-size:1rem;font-weight:800;overflow-wrap:anywhere}"
    ".highlight-value{color:var(--teal);font-size:1.4rem;font-weight:800;font-variant-numeric:tabular-nums}"
    ".highlight-note{color:var(--muted);font-size:.72rem}"
    ".changes{padding-top:4px}.change-list{margin:10px 0 0;padding-left:20px;color:var(--muted);font-size:.88rem}"
    ".whats-new{padding-top:4px}"
    ".change-grid{display:grid;grid-template-columns:1fr;gap:12px;margin-top:12px}"
    ".change-card{display:flex;flex-direction:column;gap:8px;padding:14px 16px;border:1px solid var(--line);"
    "border-left-width:4px;border-radius:14px;background:var(--paper);box-shadow:var(--shadow)}"
    ".change-card.change-new,.change-card.change-first,.change-card.change-new-window{border-left-color:var(--teal)}"
    ".change-card.change-leaderboard{border-left-color:var(--amber)}"
    ".change-card.change-quiet{border-left-color:var(--line)}"
    ".change-head{display:flex;align-items:center;justify-content:space-between;gap:8px}"
    ".change-badge{color:var(--muted);font-size:.66rem;font-weight:800;letter-spacing:.08em;text-transform:uppercase}"
    ".change-count{display:inline-grid;place-items:center;min-width:22px;height:22px;padding:0 6px;border-radius:7px;"
    "background:var(--mint);color:#096c70;font-size:.72rem;font-weight:800}"
    ".change-detail{margin:0;color:var(--ink);font-size:.92rem;line-height:1.45}"
    ".change-chips{display:flex;flex-wrap:wrap;gap:8px}"
    ".change-chip{display:inline-block;max-width:100%;padding:6px 13px;border:1px solid var(--line);"
    "border-radius:12px;background:var(--surface-muted);color:var(--ink);font-size:.84rem;line-height:1.45;"
    "overflow-wrap:anywhere}"
    "a.change-chip{text-decoration:none;cursor:pointer;transition:border-color 150ms ease,color 150ms ease,background-color 150ms ease}"
    "a.change-chip:hover,a.change-chip:focus-visible{border-color:var(--teal);color:var(--teal);background:var(--mint)}"
    ".change-chip .external-icon{margin-left:6px;color:var(--teal);font-size:.78rem;font-weight:900}"
    "section{margin:34px 0}section>header{display:flex;align-items:end;justify-content:space-between;gap:16px;margin-bottom:14px}"
    "h2{margin:0;color:var(--navy);font-size:1.35rem;letter-spacing:-.02em}h3{margin:0;color:var(--navy);font-size:1.05rem}"
    ".section-note{margin:4px 0 0;color:var(--muted);font-size:.88rem}"
    ".decision-filters{display:flex;align-items:center;flex-wrap:wrap;gap:12px;margin-top:14px;padding:8px;"
    "border:1px solid var(--line);border-radius:12px;background:var(--paper)}"
    ".filter-field{display:flex;align-items:center;gap:8px;color:var(--navy);font-size:.82rem;font-weight:750}"
    ".filter-select{min-width:190px;padding:8px 30px 8px 10px;border:1px solid var(--line);border-radius:8px;"
    "background:var(--control);color:var(--ink);font:inherit}"
    ".filter-select:hover{border-color:var(--teal);background:var(--hover)}"
    ".filter-check{display:inline-flex;align-items:center;gap:7px;color:var(--navy);font-size:.82rem;font-weight:750;cursor:pointer}"
    ".filter-check input{width:16px;height:16px;accent-color:var(--teal)}"
    ".filter-empty{display:none;padding:14px;border-radius:12px;background:var(--surface-muted);color:var(--muted);font-size:.85rem}"
    ".warnings{display:grid;gap:8px;margin:20px 0}.warning{display:flex;gap:10px;align-items:flex-start;"
    "padding:12px 14px;border:1px solid #f1d28b;border-radius:12px;background:var(--amber-bg);color:#704300;font-size:.88rem}"
    ".warning-mark{font-weight:900;color:var(--amber)}"
    ".tabs{background:var(--paper);border:1px solid var(--line);border-radius:18px;padding:8px;box-shadow:var(--shadow)}"
    ".tab-input{position:absolute;opacity:0;pointer-events:none}"
    ".tab-labels{display:flex;gap:6px;padding:4px;border-bottom:1px solid var(--line);overflow-x:auto}"
    ".tab-label{flex:1 0 auto;padding:11px 14px;border-radius:10px;color:var(--muted);font-size:.82rem;font-weight:800;cursor:pointer;text-align:center}"
    ".tab-label:hover{background:var(--hover);color:var(--ink)}"
    ".tab-input:checked+.tab-label{background:var(--teal);color:var(--on-accent)}"
    ".tab-panel{display:none;border:0;box-shadow:none;padding:22px 10px 12px}.tab-panel.is-active{display:block}"
    ".ranking-card{min-width:0;background:var(--paper);border:1px solid var(--line);border-radius:18px;"
    "padding:18px;box-shadow:var(--shadow)}.ranking-card .metric{min-height:42px;margin:8px 0 16px;color:var(--muted);font-size:.78rem}"
    ".ranking-card.unavailable{background:var(--surface-muted)}.unavailable-note{padding:14px;border-radius:12px;background:var(--rose-bg);color:var(--rose);font-size:.85rem}"
    ".table-scroll{max-width:100%;overflow-x:auto;-webkit-overflow-scrolling:touch;overscroll-behavior-x:contain}"
    ".table-scroll table{width:100%}"
    "table{border-collapse:separate;border-spacing:0;font-size:.84rem}th{padding:9px 10px;white-space:nowrap;"
    "color:var(--muted);font-size:.68rem;font-weight:800;letter-spacing:.07em;text-align:left;text-transform:uppercase;"
    "border-bottom:1px solid var(--line)}td{padding:11px 10px;border-bottom:1px solid var(--row-line);vertical-align:top;white-space:nowrap}"
    "tbody tr:last-child td{border-bottom:0}tbody tr:hover{background:var(--hover)}"
    ".decision-row[data-open-weight='true'] td{background:var(--open-row)}.decision-row[data-open-weight='true']:hover td{background:var(--open-row-hover)}"
    ".rank{display:inline-grid;place-items:center;width:26px;height:26px;border-radius:8px;"
    "background:var(--mint);color:#096c70;font-size:.75rem;font-weight:800}"
    ".model{display:block;max-width:220px;color:var(--ink);font-weight:750;overflow-wrap:anywhere;"
    "white-space:normal;min-width:150px}"
    ".open-weight-mark{display:inline-block;margin-left:6px;padding:2px 5px;border:1px solid color-mix(in srgb,var(--teal) 58%,transparent);border-radius:5px;color:var(--teal);font-size:.62rem;font-weight:800;letter-spacing:.04em;vertical-align:middle}"
    ".sub{display:block;margin-top:2px;color:var(--muted);font-size:.72rem;white-space:normal}"
    ".number{font-variant-numeric:tabular-nums;white-space:nowrap}.source-link{color:var(--teal);font-weight:750;text-decoration:none}"
    ".source-link{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border:1px solid var(--link-border);border-radius:8px;background:var(--link-bg)}"
    ".source-link:hover{background:var(--mint);border-color:var(--teal);text-decoration:none}.external-icon{font-size:1.05rem;font-weight:900;line-height:1}.unknown{color:#98a2b3}"
    ".sort-button{display:inline-flex;align-items:center;gap:5px;border:0;padding:0;background:transparent;color:inherit;font:inherit;font-size:inherit;font-weight:inherit;letter-spacing:inherit;text-align:left;text-transform:inherit;cursor:pointer}"
    ".sort-button:hover{color:var(--teal)}.sort-button:after{content:'↕';color:#98a2b3;font-size:.8rem}.sort-button[aria-sort='ascending']:after{content:'↑';color:var(--teal)}.sort-button[aria-sort='descending']:after{content:'↓';color:var(--teal)}"
    "button:focus-visible,select:focus-visible{outline:2px solid var(--teal);outline-offset:2px}"
    ".footer-note{margin:30px 0 0;color:var(--muted);font-size:.78rem}"
    "@media(max-width:620px){.page{padding:16px 12px 36px}section{margin:26px 0}.ranking-card{padding:18px 14px}"
    ".ranking-card .metric{min-height:0;margin:7px 0 12px}.ranking-card th,.ranking-card td{padding:10px 8px}}"
)
_STYLE_HASH = base64.b64encode(hashlib.sha256(_STYLE.encode("utf-8")).digest()).decode("ascii")
_SCRIPT = """(function(){
function sortableValue(value){
    var text=value.trim();
    if(!text||/^(unknown|n\\/a|--|unavailable)$/i.test(text))return{missing:true};
    var numeric=text.replace(/[$,%]/g,'').replace(/,/g,'');
    if(/^-?\\d+(\\.\\d+)?$/.test(numeric))return{number:Number(numeric)};
    var date=Date.parse(text);
    if(!Number.isNaN(date)&&/[-:]|T/.test(text))return{number:date};
    return{string:text.toLocaleLowerCase()};
}
function sortTable(table,index,button){
    var body=table.tBodies[0];
    if(!body)return;
    var descending=button.dataset.direction==='ascending';
    table.querySelectorAll('.sort-button').forEach(function(item){item.dataset.direction='none';item.setAttribute('aria-sort','none');});
    button.dataset.direction=descending?'descending':'ascending';
    button.setAttribute('aria-sort',descending?'descending':'ascending');
    var rows=Array.from(body.rows).map(function(row,position){return{row:row,position:position,value:sortableValue(row.cells[index]?.textContent||'')}});
    rows.sort(function(left,right){
        if(left.value.missing!==right.value.missing)return left.value.missing?1:-1;
        var result=left.value.number!==undefined&&right.value.number!==undefined?left.value.number-right.value.number:(left.value.string||'').localeCompare(right.value.string||'',undefined,{numeric:true,sensitivity:'base'});
        return result===0?left.position-right.position:(descending?-result:result);
    });
    rows.forEach(function(item){body.appendChild(item.row);});
}
document.querySelectorAll('table').forEach(function(table){
    var headers=table.querySelectorAll('thead th');
    headers.forEach(function(header,index){
        var label=header.textContent.trim();
        var button=document.createElement('button');
        button.type='button';button.className='sort-button';button.textContent=label;button.dataset.direction='none';button.setAttribute('aria-sort','none');button.setAttribute('aria-label','Sort by '+label);
        button.addEventListener('click',function(){sortTable(table,index,button);});
        header.textContent='';header.appendChild(button);
    });
});
function applyTheme(theme){
    var light=theme==='light';
    document.body.dataset.theme=light?'light':'dark';
    var button=document.getElementById('theme-toggle');
    if(button){
        button.textContent=light?'☾':'☼';
        button.setAttribute('aria-label',light?'Use dark theme':'Use light theme');
        button.title=light?'Use dark theme':'Use light theme';
    }
}
applyTheme('dark');
document.getElementById('theme-toggle')?.addEventListener('click',function(){
    var theme=document.body.dataset.theme==='light'?'dark':'light';
    applyTheme(theme);
});
function applyDecisionFilters(){
    var select=document.getElementById('decision-modality');
    var type=select?.value||'llm';
    var openOnly=document.getElementById('decision-open-weight')?.checked||false;
    var tabInputs=Array.from(document.querySelectorAll('.tab-input'));
    tabInputs.forEach(function(input){
        var allowed=(input.dataset.modelTypes||'')===type;
        input.hidden=!allowed;
        var label=document.querySelector('label[for="'+input.id+'"]');
        if(label)label.hidden=!allowed;
    });
    var checked=tabInputs.find(function(input){return input.checked&&!input.hidden;});
    if(!checked){
        var firstAllowed=tabInputs.find(function(input){return !input.hidden;});
        if(firstAllowed)firstAllowed.checked=true;
    }
    document.querySelectorAll('.tab-panel').forEach(function(panel){
        var input=document.getElementById('primary-tab-'+panel.id.replace('primary-panel-',''));
        var active=!!input&&input.checked&&!input.hidden;
        panel.classList.toggle('is-active',active);
        if(!active)return;
        panel.querySelectorAll('.decision-table').forEach(function(table){
            var rows=Array.from(table.querySelectorAll('.decision-row'));
            var count=0;
            rows.forEach(function(row){
                var matches=!openOnly||row.dataset.openWeight==='true';
                row.hidden=!matches||count>=10;
                if(matches)count+=1;
            });
            var visible=rows.filter(function(row){return !row.hidden;});
            visible.forEach(function(row,index){
                var rank=row.querySelector('.rank');
                if(rank)rank.textContent=String(index+1);
            });
        });
        var empty=panel.querySelector('.filter-empty');
        if(empty){
            var table=panel.querySelector('.decision-table');
            var hasRows=!!table&&table.querySelectorAll('.decision-row').length>0;
            var visibleRows=table?table.querySelectorAll('.decision-row:not([hidden])').length:0;
            empty.style.display=hasRows&&!visibleRows?'block':'none';
        }
    });
}
document.getElementById('decision-modality')?.addEventListener('change',applyDecisionFilters);
document.getElementById('decision-open-weight')?.addEventListener('change',applyDecisionFilters);
document.querySelectorAll('.tab-input').forEach(function(input){
    input.addEventListener('change',applyDecisionFilters);
});
document.querySelectorAll('.highlight').forEach(function(card){
    card.addEventListener('click',function(){
        var index=card.dataset.gotoTab;
        var input=document.getElementById('primary-tab-'+index);
        if(!input)return;
        var select=document.getElementById('decision-modality');
        if(select)select.value='llm';
        input.checked=true;
        applyDecisionFilters();
        document.getElementById('primary-panel-'+index)?.scrollIntoView({behavior:'smooth',block:'start'});
    });
});
applyDecisionFilters();
})();"""
_SCRIPT_HASH = base64.b64encode(hashlib.sha256(_SCRIPT.encode("utf-8")).digest()).decode("ascii")
_CSP = (
    f"default-src 'none'; img-src data:; style-src 'sha256-{_STYLE_HASH}'; "
    f"script-src 'sha256-{_SCRIPT_HASH}'; base-uri 'none'; form-action 'none'"
)

_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">'
    '<rect width="32" height="32" rx="7" fill="#0d141b"/>'
    '<g fill="none" stroke="#2f7d76" stroke-width="1.2">'
    '<circle cx="16" cy="16" r="9.8"/>'
    '<circle cx="16" cy="16" r="13.2"/>'
    "</g>"
    '<path d="M16 16 L16 2.8 A13.2 13.2 0 0 1 26.4 8.6 L16 16 Z" fill="#66d5c5" opacity=".18"/>'
    '<line x1="16" y1="16" x2="26.4" y2="8.6" stroke="#66d5c5" stroke-width="1.9" '
    'stroke-linecap="round"/>'
    '<circle cx="22" cy="11.4" r="1.9" fill="#66d5c5"/>'
    '<circle cx="10.6" cy="20" r="1.2" fill="#66d5c5" opacity=".55"/>'
    '<circle cx="16" cy="16" r="1.7" fill="#66d5c5"/>'
    "</svg>"
)
_FAVICON = "data:image/svg+xml;base64," + base64.b64encode(_FAVICON_SVG.encode("utf-8")).decode(
    "ascii"
)

_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


FP16_BYTES_PER_PARAMETER = 2.0


def format_size_gb(parameters_b: float | None) -> str:
    """Estimated weights size in GB at FP16 (2 bytes per parameter)."""
    if parameters_b is None:
        return "unknown"
    return f"{parameters_b * FP16_BYTES_PER_PARAMETER:.1f}"


def format_date(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, datetime):
        current = value
    elif isinstance(value, date):
        current = datetime(value.year, value.month, value.day, tzinfo=UTC)
    elif isinstance(value, str):
        try:
            current = datetime.fromisoformat(value)
        except ValueError:
            return value
    else:
        return str(value)
    return f"{current.day:02d} {_MONTHS[current.month - 1]} {current.year:04d}"


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Radar</title>
<link rel="icon" type="image/svg+xml" href="{{ favicon }}">
<meta http-equiv="Content-Security-Policy" content="{{ csp }}">
<style>{{ style|safe }}</style>
</head>
<body data-theme="dark">
<main class="page">
<p class="report-meta"><span>{{ snapshot.status|title }} · Generated {{ format_date(snapshot.generated_at) }}</span><button class="theme-toggle" id="theme-toggle" type="button" aria-label="Use light theme" title="Use light theme">☼</button></p>
{% if highlights %}
<section class="highlights" aria-label="Top picks">
{% for card in highlights %}
<button type="button" class="highlight" data-goto-tab="{{ card.tab_index }}" aria-label="{{ card.label }}: {{ card.model.name }}, {{ card.value }}">
<span class="highlight-label">{{ card.label }}</span>
<span class="highlight-model">{{ card.model.name }}{% if card.model.open_weights is true %}<span class="open-weight-mark" title="Open weights" aria-label="Open weights">OW</span>{% endif %}</span>
<span class="highlight-value">{{ card.value }}</span>
<span class="highlight-note">{{ card.note }}</span>
</button>
{% endfor %}
</section>
{% endif %}
{% set change_items = snapshot.changes.get('items') %}
{% if change_items %}<section class="whats-new" aria-label="What's new"><header><div><h2>What's new</h2><p class="section-note">Since the previous retained snapshot.</p></div></header><div class="change-grid">{% for item in change_items %}<article class="change-card change-{{ item.kind }}"><div class="change-head"><span class="change-badge">{{ item.label }}</span>{% if item.count %}<span class="change-count">{{ item.count }}</span>{% endif %}</div><p class="change-detail">{{ item.detail }}</p>{% if item.examples %}<div class="change-chips">{% for example in item.examples %}{% set example_text = example.text if example is mapping and example.text else example %}{% set example_url = safe_url(example.url) if example is mapping else None %}{% if example_url %}<a class="change-chip change-chip-link" href="{{ example_url }}" target="_blank" rel="noopener noreferrer" title="Open source for {{ example_text }}">{{ example_text }}<span class="external-icon" aria-hidden="true">↗</span></a>{% else %}<span class="change-chip">{{ example_text }}</span>{% endif %}{% endfor %}</div>{% endif %}</article>{% endfor %}</div></section>
{% elif snapshot.changes.get('summary') %}<section class="changes" aria-label="What's new"><header><div><h2>What's new</h2><p class="section-note">Since the previous retained snapshot.</p></div></header><ul class="change-list">{% for change in snapshot.changes.get('summary', []) %}<li>{{ change }}</li>{% endfor %}</ul></section>{% endif %}
<section><header><div><h2>Decision views</h2><p class="section-note">Shortlists for choosing what deserves attention now.</p></div></header>
<div class="decision-filters" aria-label="Decision view filters">
<label class="filter-field" for="decision-modality">Model type
<select class="filter-select" id="decision-modality">
{% for group in model_type_groups(model_type_tabs) %}<option value="{{ group }}" data-categories="{{ model_type_group_categories(group)|join(',') }}"{% if loop.first %} selected{% endif %}>{{ model_type_group_label(group) }}</option>{% endfor %}
</select></label>
<label class="filter-check"><input id="decision-open-weight" type="checkbox"> Open-weight only</label>
</div>
<div class="tabs">
<div class="tab-labels">{% for tab in primary_tabs %}<input class="tab-input" type="radio" name="primary-view" id="primary-tab-{{ loop.index0 }}" data-model-types="{{ tab.group }}"{% if loop.first %} checked{% endif %}><label class="tab-label" for="primary-tab-{{ loop.index0 }}" data-model-types="{{ tab.group }}">{{ tab.title }}</label>{% endfor %}</div>
<div class="tab-panels">
{% for tab in primary_tabs %}
<article id="primary-panel-{{ loop.index0 }}" data-model-types="{{ tab.group }}" class="ranking-card tab-panel{{ ' is-active' if loop.first else '' }}{{ ' unavailable' if tab.unavailable else '' }}"><h3>{{ tab.title }}</h3>
{% if tab.unavailable %}
<p class="unavailable-note">{{ tab.view.annotations.get('reason', 'unavailable') }}</p>
{% else %}
<p class="metric">{{ tab.view.annotations.get('metric', tab.view.annotations.get('heuristic', '')) }}</p>
<div class="table-scroll"><table class="decision-table" data-model-table="{{ tab.category }}"><thead><tr><th>Rank</th><th>Model</th>
{% if tab.category != 'llm' %}
<th>AA Elo</th><th>API cost</th><th>Samples</th><th>Released</th><th>Open weights</th>
{% elif tab.view.view_id == 'tiny-llm-top10' %}
<th>Params (B)</th><th>Size (GB)</th><th>Benchmark (AA Index)</th><th>LiveBench</th><th>Cost per benchmark task (USD)</th><th>Median output tokens/s</th>
{% elif tab.view.view_id == 'mini-llm-top10' %}
<th>Params (B)</th><th>Size (GB)</th><th>Benchmark (AA Index)</th><th>LiveBench</th><th>Downloads</th><th>HF created/updated</th>
{% elif tab.view.view_id == 'edge-models-top10' %}
<th>Params (B)</th><th>Size (GB)</th><th>Benchmark (AA Index)</th><th>LiveBench</th><th>Downloads</th><th>HF created/updated</th>
{% elif tab.view.view_id == 'performance-top5' %}
<th>LiveBench</th><th>AA Intelligence Index</th><th>Cost per benchmark task (USD)</th><th>Median output tokens/s</th>
{% elif tab.view.view_id == 'performance-per-token-top5' %}
<th>LiveBench</th><th>AA Intelligence / weighted USD per 1M tokens</th><th>AA Intelligence Index</th><th>Input USD / 1M</th><th>Output USD / 1M</th>
{% elif tab.view.view_id == 'org-copilot-per-token-top10' %}
<th>LiveBench</th><th>AA Intelligence / Copilot credits</th><th>AA Intelligence Index</th><th>Copilot credits In / Out</th><th>Thinking level</th>
{% elif tab.view.view_id == 'org-copilot-best-top10' %}
<th>LiveBench</th><th>AA Intelligence Index</th><th>AA Intelligence / Copilot credits</th><th>Copilot credits In / Out</th><th>Thinking level</th>
{% elif tab.view.view_id == 'benchmark-synthesis-top10' %}
<th>LiveBench</th><th>EvalPlus</th><th>DeepSWE</th><th>Merged index</th><th>Coverage</th><th>AA Intelligence</th>
{% else %}
<th>HF created/updated</th><th>Downloads</th><th>Likes</th><th>Parameters (B)</th>
{% endif %}
{% if tab.category == 'llm' and tab.view.view_id not in ('meaningful-new-hf-top5', 'edge-models-top10') %}<th>Copilot</th>{% endif %}<th>Source</th></tr></thead><tbody>
{% for model in tab.models %}
{% set source_value = model.source_urls[0] if tab.view.view_id in ('meaningful-new-hf-top5', 'edge-models-top10') and model.source_urls else model.artificial_analysis_model_url %}
<tr class="decision-row" data-model-type="{{ model_type(model) }}" data-open-weight="{{ 'true' if model.open_weights is true else 'false' }}"><td><span class="rank">{{ loop.index }}</span></td><td><span class="model">{{ model.name }}{% if model.open_weights is true %}<span class="open-weight-mark" title="Open weights" aria-label="Open weights">OW</span>{% endif %}</span><span class="sub">{{ model.organization or 'source metadata' }}</span></td>
{% if tab.category != 'llm' %}
<td>{{ model.artificial_analysis_modality_elo if model.artificial_analysis_modality_elo is not none else 'unknown' }}</td><td>{% if model.artificial_analysis_modality_cost is not none %}${{ model.artificial_analysis_modality_cost }} / {{ model.artificial_analysis_modality_cost_unit }}{% else %}unknown{% endif %}</td><td>{{ model.artificial_analysis_modality_samples if model.artificial_analysis_modality_samples is not none else 'unknown' }}</td><td>{{ model.artificial_analysis_modality_release or 'unknown' }}</td><td>{{ 'yes' if model.open_weights is true else 'no' if model.open_weights is false else 'unknown' }}</td>
{% elif tab.view.view_id == 'tiny-llm-top10' %}
<td>{{ model.parameters_b if model.parameters_b is not none else 'unknown' }}</td>
<td>{{ format_size_gb(model.parameters_b) }}</td>
<td>{{ model.intelligence_index if model.intelligence_index is not none else 'unknown' }}</td>
<td>{{ model.livebench_index if model.livebench_index is not none else 'unknown' }}</td>
<td>{{ model.cost_per_task_usd if model.cost_per_task_usd is not none else 'unknown' }}</td><td>{{ model.median_output_tokens_per_second if model.median_output_tokens_per_second is not none else 'unknown' }}</td>
{% elif tab.view.view_id == 'mini-llm-top10' %}
<td>{{ model.parameters_b if model.parameters_b is not none else 'unknown' }}</td>
<td>{{ format_size_gb(model.parameters_b) }}</td>
<td>{{ model.intelligence_index if model.intelligence_index is not none else 'unknown' }}</td>
<td>{{ model.livebench_index if model.livebench_index is not none else 'unknown' }}</td>
<td>{{ model.downloads if model.downloads is not none else 'unknown' }}</td>
<td>{{ format_date(model.created_at or model.updated_at) }}</td>
{% elif tab.view.view_id == 'edge-models-top10' %}
{% set bench = tab.benchmarks.get(model.model_id) %}
<td>{{ model.parameters_b if model.parameters_b is not none else 'unknown' }}</td>
<td>{{ format_size_gb(model.parameters_b) }}</td>
<td>{{ model.intelligence_index if model.intelligence_index is not none else (bench[0] if bench and bench[0] is not none else 'unknown') }}</td>
<td>{{ model.livebench_index if model.livebench_index is not none else (bench[1] if bench and bench[1] is not none else 'unknown') }}</td>
<td>{{ model.downloads if model.downloads is not none else 'unknown' }}</td>
<td>{{ format_date(model.created_at or model.updated_at) }}</td>
{% elif tab.view.view_id == 'performance-top5' %}
<td>{{ model.livebench_index if model.livebench_index is not none else 'unknown' }}</td>
<td>{{ model.intelligence_index if model.intelligence_index is not none else 'unknown' }}</td><td>{{ model.cost_per_task_usd if model.cost_per_task_usd is not none else 'unknown' }}</td><td>{{ model.median_output_tokens_per_second if model.median_output_tokens_per_second is not none else 'unknown' }}</td>
{% elif tab.view.view_id == 'performance-per-token-top5' %}
<td>{{ model.livebench_index if model.livebench_index is not none else 'unknown' }}</td>
<td>{{ model.scores.get('aa_token_dollar_efficiency').value if model.scores.get('aa_token_dollar_efficiency') and model.scores.get('aa_token_dollar_efficiency').value is not none else 'unknown' }}</td><td>{{ model.intelligence_index if model.intelligence_index is not none else 'unknown' }}</td><td>{{ model.artificial_analysis_input_price_per_million if model.artificial_analysis_input_price_per_million is not none else 'unknown' }}</td><td>{{ model.artificial_analysis_output_price_per_million if model.artificial_analysis_output_price_per_million is not none else 'unknown' }}</td>
{% elif tab.view.view_id == 'org-copilot-per-token-top10' %}
<td>{{ model.livebench_index if model.livebench_index is not none else 'unknown' }}</td>
<td>{{ model.scores.get('copilot_token_efficiency').value if model.scores.get('copilot_token_efficiency') and model.scores.get('copilot_token_efficiency').value is not none else 'unknown' }}</td><td>{{ model.intelligence_index if model.intelligence_index is not none else 'unknown' }}</td><td>{{ model.copilot_input_credits_per_million }} / {{ model.copilot_output_credits_per_million }}</td><td>{{ model.effort_level or 'default' }}</td>
{% elif tab.view.view_id == 'org-copilot-best-top10' %}
<td>{{ model.livebench_index if model.livebench_index is not none else 'unknown' }}</td>
<td>{{ model.intelligence_index if model.intelligence_index is not none else 'unknown' }}</td><td>{{ model.scores.get('copilot_token_efficiency').value if model.scores.get('copilot_token_efficiency') and model.scores.get('copilot_token_efficiency').value is not none else 'unknown' }}</td><td>{{ model.copilot_input_credits_per_million }} / {{ model.copilot_output_credits_per_million }}</td><td>{{ model.effort_level or 'default' }}</td>
{% elif tab.view.view_id == 'benchmark-synthesis-top10' %}
<td>{{ model.livebench_index if model.livebench_index is not none else 'unknown' }}</td>
<td>{{ model.evalplus_index if model.evalplus_index is not none else 'unknown' }}</td>
<td>{{ model.deepswe_index if model.deepswe_index is not none else 'unknown' }}</td>
<td>{{ model.merged_benchmark_index if model.merged_benchmark_index is not none else 'unknown' }}</td>
<td>{{ model.benchmark_coverage }}/3</td>
<td>{{ model.intelligence_index if model.intelligence_index is not none else 'unknown' }}</td>
{% else %}
<td>{{ format_date(model.created_at or model.updated_at) }}</td>
<td>{{ model.downloads if model.downloads is not none else 'unknown' }}</td>
<td>{{ model.likes if model.likes is not none else 'unknown' }}</td>
<td>{{ model.parameters_b if model.parameters_b is not none else 'unknown' }}</td>
{% endif %}
{% if tab.category == 'llm' and tab.view.view_id not in ('meaningful-new-hf-top5', 'edge-models-top10') %}<td>{{ 'yes' if model.copilot_ready is true else 'no' if model.copilot_ready is false else 'unknown' }}</td>{% endif %}
{% if source_value %}<td><a class="source-link" href="{{ safe_url(source_value) }}" title="Open source" aria-label="Open source for {{ model.name }}"><span class="external-icon" aria-hidden="true">↗</span></a></td>{% else %}<td class="unknown">unknown</td>{% endif %}
</tr>
{% endfor %}
</tbody></table></div>
<p class="filter-empty">No models match the selected filters.</p>
{% endif %}
</article>
{% endfor %}
</div></div></section>
<p class="footer-note">AA benchmark-task cost is displayed separately from token pricing. Copilot availability is shown only when an explicit availability source confirms it.</p>
</main>
<script>{{ script|safe }}</script>
</body></html>"""

_PRIMARY_VIEW_IDS = (
    "org-copilot-per-token-top10",
    "org-copilot-best-top10",
    "benchmark-synthesis-top10",
    "performance-top5",
    "performance-per-token-top5",
    "tiny-llm-top10",
    "mini-llm-top10",
    "edge-models-top10",
    "meaningful-new-hf-top5",
)

_MODEL_TYPE_LABELS = {
    "llm": "LLM",
    "text-to-image": "Text to image",
    "image-to-image": "Image to image",
    "text-to-video": "Text to video",
    "image-to-video": "Image to video",
}

_MODEL_TYPE_GROUPS = {
    "llm": ("llm",),
    "image": ("text-to-image", "image-to-image"),
    "video": ("text-to-video", "image-to-video"),
}

_MODEL_TYPE_GROUP_LABELS = {
    "llm": "LLM",
    "image": "Image",
    "video": "Video",
}

_MODALITY_ABBREVIATIONS = {
    "text-to-image": "t2i",
    "image-to-image": "i2i",
    "text-to-video": "t2v",
    "image-to-video": "i2v",
}

_DEFAULT_MODEL_TYPE_TABS = {
    "llm": list(_PRIMARY_VIEW_IDS),
    "image": ["performance-top5", "meaningful-new-hf-top5"],
    "video": ["performance-top5", "meaningful-new-hf-top5"],
}


def _group_for_key(key: str) -> str:
    if key in _MODEL_TYPE_GROUPS:
        return key
    for group, categories in _MODEL_TYPE_GROUPS.items():
        if key in categories:
            return group
    return key


def normalise_model_type_tabs(model_type_tabs: dict[str, list[str]]) -> dict[str, list[str]]:
    raw = model_type_tabs or _DEFAULT_MODEL_TYPE_TABS
    grouped: dict[str, list[str]] = {}
    for key, tabs in raw.items():
        bucket = grouped.setdefault(_group_for_key(key), [])
        for tab in tabs:
            if tab not in bucket:
                bucket.append(tab)
    ordered = {group: grouped[group] for group in _MODEL_TYPE_GROUP_LABELS if group in grouped}
    for group, tabs in grouped.items():
        ordered.setdefault(group, tabs)
    return ordered


def model_type_groups(model_type_tabs: dict[str, list[str]]) -> list[str]:
    return list(normalise_model_type_tabs(model_type_tabs))


def model_type_group_label(group: str) -> str:
    return _MODEL_TYPE_GROUP_LABELS.get(group, group)


def model_type_group_categories(group: str) -> list[str]:
    return list(_MODEL_TYPE_GROUPS.get(group, (group,)))


def _model_type_catalog(models: list[ModelRecord]) -> dict[str, list[ModelRecord]]:
    catalog: dict[str, list[ModelRecord]] = {}
    for category in _MODEL_TYPE_LABELS:
        matching = [model for model in models if model_type(model) == category]
        ranked = sorted(
            matching,
            key=lambda model: (
                _catalog_score(model) is None,
                -(_catalog_score(model) or 0.0),
                -(model.downloads or 0),
                -(model.likes or 0),
                model.name.casefold(),
            ),
        )
        open_weight = [model for model in matching if model.open_weights is True]
        catalog[category] = sorted(
            {model.model_id: model for model in [*ranked[:10], *open_weight[:10]]}.values(),
            key=lambda model: (
                _catalog_score(model) is None,
                -(_catalog_score(model) or 0.0),
                -(model.downloads or 0),
                -(model.likes or 0),
                model.name.casefold(),
            ),
        )
    return catalog


def _catalog_score(model: ModelRecord) -> float | None:
    return (
        model.artificial_analysis_modality_elo
        if model_type(model) != "llm"
        else model.intelligence_index
    )


def model_type_label(category: str) -> str:
    return _MODEL_TYPE_LABELS.get(category, category)


def view_model_types(view_id: str, model_type_tabs: dict[str, list[str]]) -> list[str]:
    groups = normalise_model_type_tabs(model_type_tabs)
    allowed = [group for group, tabs in groups.items() if view_id in tabs]
    return allowed or list(groups)


def modality_view_title(title: str, category: str) -> str:
    """Append a modality abbreviation to a tab title, e.g. "Performance t2i top 10"."""
    abbreviation = _MODALITY_ABBREVIATIONS.get(category)
    if not abbreviation:
        return title
    if "top 10" in title:
        return title.replace("top 10", f"{abbreviation} top 10")
    return f"{title} {abbreviation}"


@dataclass(frozen=True)
class _PrimaryTab:
    view: View
    category: str
    group: str
    title: str
    models: list[ModelRecord]
    unavailable: bool
    benchmarks: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)


def _benchmark_index(
    models: list[ModelRecord],
) -> dict[str, ModelRecord]:
    """Best benchmark-bearing record per normalised family name."""
    best: dict[str, ModelRecord] = {}
    for model in models:
        if model.intelligence_index is None and model.livebench_index is None:
            continue
        key = _benchmark_family_key(model.name)
        current = best.get(key)
        if current is None or (model.intelligence_index or 0) > (current.intelligence_index or 0):
            best[key] = model
    return best


def benchmark_enrichment(
    models: list[ModelRecord], index: dict[str, ModelRecord]
) -> dict[str, tuple[float | None, float | None]]:
    """Benchmark values for models that lack their own, matched by family name.

    Hugging Face edge and on-device records are distinct from their Artificial
    Analysis counterparts, so a model like ``openbmb/MiniCPM5-2B`` has no score
    of its own. This borrows the score from the matching name where one exists,
    and leaves genuinely uncovered models absent.
    """
    enriched: dict[str, tuple[float | None, float | None]] = {}
    for model in models:
        if model.intelligence_index is not None or model.livebench_index is not None:
            continue
        match = index.get(_benchmark_family_key(model.name))
        if match is not None:
            enriched[model.model_id] = (match.intelligence_index, match.livebench_index)
    return enriched


def primary_tabs(snapshot: Snapshot, model_type_tabs: dict[str, list[str]]) -> list[_PrimaryTab]:
    """One tab per (view, modality), grouped by model type.

    LLM views yield a single tab; image and video views yield one tab per
    modality (text to image / image edit, text to video / image to video).
    """
    model_by_id = {model.model_id: model for model in snapshot.models}
    catalog = _model_type_catalog(snapshot.models)
    benchmark_index = _benchmark_index(snapshot.models)
    tabs: list[_PrimaryTab] = []
    for view in snapshot.views:
        if view.view_id not in _PRIMARY_VIEW_IDS:
            continue
        groups = view_model_types(view.view_id, model_type_tabs)
        by_category = decision_models_by_type(view, model_by_id, catalog)
        emitted = False
        for group in groups:
            for category in model_type_group_categories(group):
                models = by_category.get(category, [])
                if not models:
                    continue
                benchmarks = (
                    benchmark_enrichment(models, benchmark_index)
                    if view.view_id == "edge-models-top10"
                    else {}
                )
                tabs.append(
                    _PrimaryTab(
                        view=view,
                        category=category,
                        group=group,
                        title=modality_view_title(view.title, category),
                        models=models,
                        unavailable=False,
                        benchmarks=benchmarks,
                    )
                )
                emitted = True
        if not emitted:
            group = groups[0] if groups else "llm"
            category = model_type_group_categories(group)[0]
            tabs.append(
                _PrimaryTab(
                    view=view,
                    category=category,
                    group=group,
                    title=modality_view_title(view.title, category),
                    models=[],
                    unavailable=True,
                )
            )
    return tabs


_HIGHLIGHT_VIEWS = (
    ("performance-top5", "Best power LLM", "AA Intelligence Index"),
    (
        "performance-per-token-top5",
        "Best value per token",
        "AA Intelligence per weighted USD per 1M tokens",
    ),
    ("tiny-llm-top10", "Best tiny LLM", "AA Intelligence Index at or below 8B parameters"),
    ("mini-llm-top10", "Best mini LLM", "AA Intelligence Index at or below 1B parameters"),
)


@dataclass(frozen=True)
class _Highlight:
    label: str
    note: str
    value: str
    model: ModelRecord
    tab_index: int


def _highlight_value(view_id: str, model: ModelRecord) -> str:
    if view_id == "performance-per-token-top5":
        score = model.scores.get("aa_token_dollar_efficiency")
        return f"{score.value:.2f}" if score and score.value is not None else "unknown"
    index = model.intelligence_index
    return f"{index:g}" if index is not None else "unknown"


def highlights(tabs: list[_PrimaryTab]) -> list[_Highlight]:
    """Top LLM pick for each headline view, for the summary cards at the top."""
    cards: list[_Highlight] = []
    for view_id, label, note in _HIGHLIGHT_VIEWS:
        for index, tab in enumerate(tabs):
            if tab.view.view_id != view_id or tab.group != "llm" or not tab.models:
                continue
            model = tab.models[0]
            cards.append(
                _Highlight(
                    label=label,
                    note=note,
                    value=_highlight_value(view_id, model),
                    model=model,
                    tab_index=index,
                )
            )
            break
    return cards


def decision_models(
    view: object,
    model_by_id: dict[str, ModelRecord],
    model_type_models: dict[str, list[ModelRecord]],
) -> list[ModelRecord]:
    ids = list(getattr(view, "model_ids", []))
    for group in model_type_models.values():
        ids.extend(model.model_id for model in group)
    seen: set[str] = set()
    models: list[ModelRecord] = []
    for model_id in ids:
        if model_id in model_by_id and model_id not in seen:
            seen.add(model_id)
            models.append(model_by_id[model_id])
    return models


def decision_models_by_type(
    view: object,
    model_by_id: dict[str, ModelRecord],
    model_type_models: dict[str, list[ModelRecord]],
) -> dict[str, list[ModelRecord]]:
    joined = decision_models(view, model_by_id, model_type_models)
    return {
        category: [model for model in joined if model_type(model) == category]
        for category in _MODEL_TYPE_LABELS
    }


def render_html(snapshot: Snapshot) -> bytes:
    environment = Environment(loader=BaseLoader(), autoescape=select_autoescape(["html", "xml"]))
    environment.globals["safe_url"] = _safe_url
    model_type_tabs = snapshot.summary.get("model_type_tabs", _DEFAULT_MODEL_TYPE_TABS)
    tabs = primary_tabs(snapshot, model_type_tabs)
    return (
        environment.from_string(_TEMPLATE)
        .render(
            snapshot=snapshot,
            csp=_CSP,
            favicon=_FAVICON,
            format_date=format_date,
            format_size_gb=format_size_gb,
            style=_STYLE,
            script=_SCRIPT,
            primary_tabs=tabs,
            highlights=highlights(tabs),
            model_type=model_type,
            model_type_tabs=model_type_tabs,
            view_model_types=view_model_types,
            model_type_groups=model_type_groups,
            model_type_group_label=model_type_group_label,
            model_type_group_categories=model_type_group_categories,
        )
        .encode("utf-8")
    )


def _safe_url(value: object) -> str | None:
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        return value
    return None
