import json, os, html
D = os.path.dirname(os.path.abspath(__file__))
r1 = json.load(open(f"{D}/results.json"))
r2 = json.load(open(f"{D}/results2.json"))
r3 = json.load(open(f"{D}/results3.json"))
r5 = json.load(open(f"{D}/results5.json"))
CW = r1["clean_w"]
by = {c["name"]: c for c in r1["cases"]}

def img(b64, w_obs, cls=""):
    pct = 100.0 * w_obs / CW
    return (f'<img class="strip {cls}" style="width:{pct:.1f}%" '
            f'src="data:image/png;base64,{b64}" alt="">')

def profile_svg(prof, thresh=None, w=760, h=64):
    n = len(prof)
    pts = " ".join(f"{i*w/(n-1):.1f},{h-(v*h):.1f}" for i, v in enumerate(prof))
    tl = ""
    if thresh:
        y = h - thresh * h
        tl = (f'<line x1="0" y1="{y:.1f}" x2="{w}" y2="{y:.1f}" class="thr"/>'
              f'<text x="{w-2}" y="{y-4:.1f}" class="thrlab">판독 하한</text>')
    return (f'<svg class="prof" viewBox="0 0 {w} {h}" preserveAspectRatio="none" '
            f'role="img" aria-label="압축률 프로파일">'
            f'<line x1="0" y1="0" x2="{w}" y2="0" class="one"/>{tl}'
            f'<polyline points="{pts}" class="m"/></svg>')

def verdict(t):
    return (f'<span class="v ok">읽힘</span>' if t else '<span class="v no">실패</span>')

# ---- 표본판 -------------------------------------------------------------
SPECIMENS = [
    ("cyl0",  "왜곡 없음"),
    ("cyl35", "원통 35°"),
    ("cyl70", "원통 70°"),
    ("sine_weak", "물결 (완만)"),
    ("crease", "접힘 α=55°"),
    ("ridge30", "접힘 + 능선각 30°"),
    ("crumple", "구김 3옥타브"),
    ("mixed", "원통 45° + 물결 + 구김"),
]
spec_html = []
for name, title in SPECIMENS:
    c = by[name]
    thr = 2.0 / 5.5
    spec_html.append(f'''
<figure class="plate">
  <figcaption>
    <h3>{html.escape(title)}</h3>
    <dl class="meta">
      <div><dt>관측 폭</dt><dd>{c["w_obs"]}px<span class="of"> / {CW}</span></dd></div>
      <div><dt>최소 모듈</dt><dd>{c["mod_min_px"]}px</dd></div>
      <div><dt>1차 디코딩</dt><dd>{verdict(c["decode_first"])}</dd></div>
      <div><dt>보정 후</dt><dd>{verdict(c["decode_rect"])}</dd></div>
    </dl>
  </figcaption>
  <div class="specimen">
    {img(c["obs_png"], c["w_obs"])}
    <div class="profwrap" style="width:{100.0*c["w_obs"]/CW:.1f}%">{profile_svg(c["profile"], thr)}</div>
  </div>
</figure>''')

# ---- 방향 검증 ----------------------------------------------------------
d = r2["direction"]
def curve(vals, cls, w=760, h=180):
    n = len(vals)
    return f'<polyline class="{cls}" points="' + " ".join(
        f"{i*w/(n-1):.1f},{h-v*h:.1f}" for i, v in enumerate(vals)) + '"/>'
dir_svg = (f'<svg viewBox="0 0 760 180" class="cmp" role="img" '
           f'aria-label="관측에서 펴진 좌표로 가는 매핑 세 가지 비교">'
           + curve(d["prof_analytic"], "analytic")
           + curve(d["prof_right"], "right")
           + curve(d["prof_wrong"], "wrong") + '</svg>')

# ---- 12점 한계 ----------------------------------------------------------
sine_rows = "".join(
    f'<tr><td class="mono">{r["lam_frac"]:.2f} W</td>'
    f'<td class="mono">{r["m_min"]:.3f}</td>'
    f'<td>{"O" if r["first"] else "·"}</td>'
    + "".join(f'<td class="{"y" if r["by_nx"][k] else "n"}">'
              f'{"읽힘" if r["by_nx"][k] else "실패"}</td>' for k in ("4","6","8","12"))
    + "</tr>" for r in r2["sine_sweep"])

pair = {i["name"]: i for i in r3["images"]}
def pairblock(key, title, note):
    p = pair[key]
    cells = "".join(
        f'<div class="rcell"><span class="rlab">{k}점 격자로 보정</span>'
        f'{img(v["png"], CW)}<span class="rres">{verdict(v["text"])}</span></div>'
        for k, v in p["rect"].items())
    return f'''<div class="pairbox">
  <h4>{html.escape(title)}</h4>
  <p class="note">{note}</p>
  <div class="obsline">{img(p["obs_png"], p["w_obs"])}</div>
  <div class="profwrap" style="width:{100.0*p["w_obs"]/CW:.1f}%">{profile_svg(p["profile"], 2.0/5.5)}</div>
  <div class="rgrid">{cells}</div>
</div>'''

# ---- d_m0 x theta 격자 --------------------------------------------------
th_list = [c["theta"] for c in r5["grid"][0]["cells"]]
grid_head = "".join(f"<th>{t}°</th>" for t in th_list)
grid_rows = ""
for row in r5["grid"]:
    tds = "".join(
        f'<td class="cell {"y" if c["rect"] else "n"}">'
        f'<span class="big">{"읽힘" if c["rect"] else "실패"}</span>'
        f'<span class="sub">{c["mod_min"]:.1f}px{"" if c["first"] else " · 1차X"}</span></td>'
        for c in row["cells"])
    grid_rows += (f'<tr><th class="mono">{row["d_m0"]}px<span class="of"> / {row["clean_w"]}</span></th>{tds}</tr>')

PAGE = f'''<title>구겨진 바코드 합성 표본판</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700&family=IBM+Plex+Mono:wght@400;500&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">
<style>
:root {{
  --ground:#FBFAF6; --surface:#FFFFFF; --sunk:#F3F1EB;
  --ink:#17150F; --body:#38342C; --muted:#726C60; --rule:#E2DDD1;
  --accent:#1F4FA8; --accent-soft:#E6ECF9;
  --pass:#1B6E3B; --fail:#A32A1C; --thr:#B8791F;
  --shadow:0 1px 2px rgba(23,21,15,.05), 0 8px 24px -16px rgba(23,21,15,.28);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#131519; --surface:#1A1D23; --sunk:#22262E;
    --ink:#EDEAE3; --body:#C9C5BC; --muted:#8E8A81; --rule:#2E333B;
    --accent:#89ABF5; --accent-soft:#1E2838;
    --pass:#5FC583; --fail:#F0897C; --thr:#DCA94F;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
  }}
}}
:root[data-theme="dark"] {{
  --ground:#131519; --surface:#1A1D23; --sunk:#22262E;
  --ink:#EDEAE3; --body:#C9C5BC; --muted:#8E8A81; --rule:#2E333B;
  --accent:#89ABF5; --accent-soft:#1E2838;
  --pass:#5FC583; --fail:#F0897C; --thr:#DCA94F;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 8px 24px -16px rgba(0,0,0,.7);
}}
* {{ box-sizing:border-box; }}
body {{
  background:var(--ground); color:var(--body); margin:0;
  font-family:"Source Serif 4",Georgia,serif; font-size:17px; line-height:1.65;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:920px; margin:0 auto; padding:64px 24px 96px; }}
h1,h2,h3,h4,.lab,th,.v,.rlab {{ font-family:Archivo,"Helvetica Neue",Arial,sans-serif; }}
h1 {{ font-size:2.5rem; line-height:1.1; font-weight:700; letter-spacing:-.02em;
     color:var(--ink); margin:0 0 .5rem; text-wrap:balance; }}
.deck {{ font-size:1.2rem; color:var(--muted); margin:0 0 8px; max-width:60ch; }}
.stamp {{ font-family:"IBM Plex Mono",monospace; font-size:.75rem; color:var(--muted);
  letter-spacing:.06em; text-transform:uppercase; }}
h2 {{ font-size:1.5rem; font-weight:600; color:var(--ink); letter-spacing:-.01em;
     margin:72px 0 4px; padding-top:20px; border-top:2px solid var(--ink); }}
h2 .lab {{ display:block; font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
  color:var(--accent); font-weight:600; margin-bottom:10px; }}
h3 {{ font-size:1.02rem; font-weight:600; color:var(--ink); margin:0; letter-spacing:-.005em; }}
h4 {{ font-size:.95rem; font-weight:600; color:var(--ink); margin:0 0 2px; }}
p {{ max-width:66ch; }}
.mono, code {{ font-family:"IBM Plex Mono",monospace; font-variant-numeric:tabular-nums; }}
code {{ background:var(--sunk); padding:.12em .38em; border-radius:3px; font-size:.86em; }}
.note {{ color:var(--muted); font-size:.92rem; margin:0 0 14px; max-width:64ch; }}
.of {{ color:var(--muted); font-weight:400; }}

/* 표본판 */
.plates {{ display:flex; flex-direction:column; gap:26px; margin-top:32px; }}
.plate {{ margin:0; background:var(--surface); border:1px solid var(--rule);
  border-radius:6px; padding:18px 20px 20px; box-shadow:var(--shadow); }}
.plate figcaption {{ display:flex; flex-wrap:wrap; align-items:baseline;
  justify-content:space-between; gap:12px; margin-bottom:14px; }}
.meta {{ display:flex; gap:20px; margin:0; flex-wrap:wrap; }}
.meta div {{ display:flex; gap:6px; align-items:baseline; }}
.meta dt {{ font-family:Archivo,sans-serif; font-size:.7rem; letter-spacing:.08em;
  text-transform:uppercase; color:var(--muted); }}
.meta dd {{ margin:0; font-family:"IBM Plex Mono",monospace; font-size:.82rem;
  color:var(--ink); font-variant-numeric:tabular-nums; }}
.specimen {{ background:var(--sunk); border-radius:4px; padding:14px; overflow-x:auto; }}
.strip {{ display:block; height:auto; image-rendering:auto; border-radius:2px;
  background:#fff; box-shadow:0 0 0 1px var(--rule); }}
.profwrap {{ margin-top:6px; }}
.prof {{ display:block; width:100%; height:56px; overflow:visible; }}
.prof .m {{ fill:none; stroke:var(--accent); stroke-width:2.5; vector-effect:non-scaling-stroke; }}
.prof .one {{ stroke:var(--rule); stroke-width:1; vector-effect:non-scaling-stroke; }}
.prof .thr {{ stroke:var(--thr); stroke-width:1.5; stroke-dasharray:5 4;
  vector-effect:non-scaling-stroke; }}
.thrlab {{ fill:var(--thr); font-family:Archivo,sans-serif; font-size:9px;
  text-anchor:end; letter-spacing:.04em; }}
.v {{ font-size:.78rem; font-weight:600; padding:.1em .5em; border-radius:3px; letter-spacing:.02em; }}
.v.ok {{ color:var(--pass); background:color-mix(in srgb, var(--pass) 12%, transparent); }}
.v.no {{ color:var(--fail); background:color-mix(in srgb, var(--fail) 12%, transparent); }}

/* 비교 그래프 */
.cmp {{ width:100%; height:200px; display:block; background:var(--sunk);
  border-radius:4px; padding:8px; }}
.cmp polyline {{ fill:none; stroke-width:2.5; vector-effect:non-scaling-stroke; }}
.cmp .analytic {{ stroke:var(--ink); stroke-width:6; opacity:.18; }}
.cmp .right {{ stroke:var(--pass); }}
.cmp .wrong {{ stroke:var(--fail); stroke-dasharray:7 5; }}
.key {{ display:flex; gap:22px; flex-wrap:wrap; margin-top:12px; font-size:.85rem; }}
.key span {{ display:flex; align-items:center; gap:8px; }}
.key i {{ width:26px; height:0; border-top-width:3px; border-top-style:solid; display:block; }}

/* 표 */
.tablewrap {{ overflow-x:auto; margin-top:20px; }}
table {{ border-collapse:collapse; width:100%; font-size:.88rem; }}
th, td {{ padding:9px 11px; text-align:left; border-bottom:1px solid var(--rule); }}
thead th {{ font-size:.7rem; letter-spacing:.09em; text-transform:uppercase;
  color:var(--muted); font-weight:600; border-bottom:1.5px solid var(--ink); }}
tbody th {{ font-family:"IBM Plex Mono",monospace; font-weight:500; color:var(--ink); }}
td.y {{ color:var(--pass); font-weight:600; }}
td.n {{ color:var(--fail); font-weight:600; }}
td.cell {{ text-align:center; }}
td.cell .big {{ display:block; font-family:Archivo,sans-serif; font-size:.82rem; font-weight:600; }}
td.cell .sub {{ display:block; font-family:"IBM Plex Mono",monospace; font-size:.7rem; color:var(--muted); }}
td.cell.y .big {{ color:var(--pass); }}
td.cell.n .big {{ color:var(--fail); }}

/* 짝 비교 */
.pairs {{ display:grid; grid-template-columns:1fr; gap:24px; margin-top:26px; }}
@media (min-width:760px) {{ .pairs {{ grid-template-columns:1fr 1fr; }} }}
.pairbox {{ background:var(--surface); border:1px solid var(--rule); border-radius:6px;
  padding:16px 18px 18px; box-shadow:var(--shadow); }}
.obsline {{ background:var(--sunk); border-radius:4px; padding:10px; }}
.rgrid {{ display:flex; flex-direction:column; gap:10px; margin-top:14px; }}
.rcell {{ display:grid; grid-template-columns:88px 1fr auto; gap:10px; align-items:center; }}
.rlab {{ font-size:.72rem; letter-spacing:.05em; text-transform:uppercase; color:var(--muted); }}
.rres {{ white-space:nowrap; }}

.callout {{ background:var(--accent-soft); border-left:3px solid var(--accent);
  border-radius:0 5px 5px 0; padding:16px 20px; margin:26px 0; }}
.callout p {{ margin:0; }}
.callout p + p {{ margin-top:10px; }}
.callout strong {{ color:var(--ink); }}
ul.find {{ list-style:none; padding:0; margin:20px 0 0; display:flex;
  flex-direction:column; gap:14px; }}
ul.find li {{ display:grid; grid-template-columns:auto 1fr; gap:14px; align-items:start;
  padding-bottom:14px; border-bottom:1px solid var(--rule); }}
ul.find b {{ font-family:"IBM Plex Mono",monospace; font-size:.72rem; letter-spacing:.04em;
  padding:.25em .5em; border-radius:3px; white-space:nowrap; margin-top:.25em; }}
.tag-fix {{ color:var(--fail); background:color-mix(in srgb, var(--fail) 12%, transparent); }}
.tag-new {{ color:var(--accent); background:var(--accent-soft); }}
.tag-ok {{ color:var(--pass); background:color-mix(in srgb, var(--pass) 12%, transparent); }}
</style>

<div class="wrap">
<header>
  <p class="stamp">WE-Meet · Stage 2 합성 설계 · 2026-08-28 측정</p>
  <h1>구겨진 바코드 합성 표본판</h1>
  <p class="deck">설계 §2의 수식이 실제로 만드는 그림과, 그것을 정답 제어점으로 되폈을 때
  읽히는지. 모델은 쓰지 않았습니다 &mdash; 정답을 그대로 넣어 잰 <em>성능의 천장</em>입니다.</p>
</header>

<h2><span class="lab">재료</span>깨끗한 Code128</h2>
<p><code>python-barcode</code>로 그린 <code>WEMEET0001</code>. 가장 얇은 막대(모듈)가 정확히
5.5px가 되게 맞추니 {CW}&times;{r1["clean_h"]}px가 나옵니다. 배경도 조명도 없습니다.</p>
<div class="specimen" style="margin-top:16px">{img(r1["clean_png"], CW)}</div>

<h2><span class="lab">표본판</span>파라미터가 만드는 그림</h2>
<p>이미지 폭이 줄어드는 것에 주의해 보십시오. <strong>감기면 정사영에서 좁아 보입니다</strong> &mdash;
관측 크롭은 펴진 폭보다 항상 좁습니다. 아래 파란 곡선은 같은 가로축 위의 압축률
<span class="mono">m</span>이고, 위쪽 회색선이 <span class="mono">m=1</span>(안 눌림),
점선이 모듈 2px에 해당하는 판독 하한입니다.</p>
<div class="plates">{"".join(spec_html)}</div>

<div class="callout">
<p><strong>막대는 세로로 곧습니다.</strong> 세로 방향으로는 아무것도 하지 않습니다(<span class="mono">v=y</span>).
사전 실험 §8.1에서 수직 사인파는 진폭 32px까지도 전부 읽혔기 때문입니다 &mdash;
정보를 파괴하는 것은 가로 압축뿐입니다.</p>
<p>그리고 지금은 음영도 반사도 없어서 <strong>&ldquo;구겨진 사진&rdquo;처럼 보이지 않습니다.</strong>
사람 눈에 구김을 알려주는 건 대부분 명암이고, 그건 §3 광학의 몫입니다.
디코더는 이 페이지의 것에만 반응합니다.</p>
</div>

<h2><span class="lab">검증 1</span>방향이 맞는지</h2>
<p>설계 문서 §4.1과 §8 파이프라인은 <code>g = cumsum(m)</code>이라고 적혀 있는데,
규약을 맞춰 유도하면 <code>cumsum(1/m)</code>이어야 합니다. 원통은 해석해
<span class="mono">r&middot;arcsin((x&minus;c)/r)</span>가 있으니 직접 대볼 수 있습니다.</p>
{dir_svg}
<div class="key">
  <span><i style="border-color:var(--ink);opacity:.35"></i>해석해 (arcsin)</span>
  <span><i style="border-color:var(--pass)"></i>cumsum(1/m) &mdash; 최대 오차 <b class="mono">{d["err_right_px"]}px</b></span>
  <span><i style="border-color:var(--fail)"></i>cumsum(m) &mdash; 최대 오차 <b class="mono">{d["err_wrong_px"]}px</b></span>
</div>
<div class="callout">
<p><strong>그런데 두 경우 모두 디코딩은 성공합니다.</strong> 잘못된 매핑으로 왜곡을 만들어도
정답 제어점을 같은 매핑에서 뽑으면 라운드트립은 자기일관적이라 읽힙니다.</p>
<p>즉 <code>cumsum(m)</code>은 <em>정답이 틀리는</em> 오류가 아니라
<strong>의도한 것과 다른 왜곡이 만들어지는</strong> 오류입니다. 디코딩 검사로는 절대 안 잡히고,
학습 분포가 현장과 어긋나 <strong>도메인 갭으로만</strong> 드러납니다. 해석해 대조가 유일한 방어선입니다.</p>
</div>

<h2><span class="lab">검증 2</span>제어점 12개로 표현되는 것과 안 되는 것</h2>
<p>같은 세기의 물결에서 <strong>파장만</strong> 바꿔가며, 가로 제어점 개수를 늘려 보정했습니다.
모든 행에서 <span class="mono">m<sub>min</sub>=0.640</span>&mdash;
<strong>정보는 충분히 살아 있습니다.</strong> 그런데도 못 읽습니다.</p>
<div class="tablewrap"><table>
<thead><tr><th>파장 λ</th><th>m<sub>min</sub></th><th>1차</th>
<th>4점 (기본안)</th><th>6점</th><th>8점</th><th>12점</th></tr></thead>
<tbody>{sine_rows}</tbody></table></div>
<p class="note">가로 제어점 <span class="mono">n<sub>x</sub></span>개가 담을 수 있는 최소 파장은
나이키스트로 <span class="mono">λ ≥ 2W/(n<sub>x</sub>&minus;1)</span> &mdash;
4점이면 0.67W, 6점이면 0.40W. 측정된 경계와 거의 정확히 맞습니다.</p>

<div class="pairs">
{pairblock("sine035", "λ = 0.35W", "같은 세기, 짧은 파장. 4점·6점으로는 못 폅니다.")}
{pairblock("sine075", "λ = 0.75W", "같은 세기, 긴 파장. 4점으로 충분하고 1차에서 이미 읽힙니다.")}
</div>

<div class="callout">
<p><strong>병목은 가로였습니다.</strong> 설계 §7.1은 세로 방향(<span class="mono">n<sub>y</sub>=3</span>)을
&ldquo;진짜 병목&rdquo;으로 지목했는데, 굽이치는 능선을 <span class="mono">λ<sub>y</sub></span>를
0.5H&ndash;4H로 바꿔가며 재보니 <span class="mono">n<sub>y</sub>=2</span>에서도 전부 읽혔습니다.
세로 변형이 무해하다는 §8.1의 성질이 여기까지 이어집니다.</p>
<p>바꿀 곳이 있다면 <strong>4&times;3이 아니라 6&times;3 또는 8&times;3</strong>입니다. 세로를 늘릴 이유는 아직 없습니다.</p>
</div>

<h2><span class="lab">검증 3</span>모듈 폭이 진짜 변수다</h2>
<p>원통 각도만으로는 좀처럼 깨지지 않아서, 촬영 해상도에 해당하는
<strong>평탄 상태 모듈 폭 <span class="mono">d<sub>m0</sub></span></strong>를 같이 흔들었습니다.
아래는 기울기 예산을 <em>끄고</em> 잰 것입니다.</p>
<div class="tablewrap"><table>
<thead><tr><th>d<sub>m0</sub> / 원본 폭</th>{grid_head}</tr></thead>
<tbody>{grid_rows}</tbody></table></div>
<p class="note">셀 = 정답 제어점(6&times;3)으로 보정한 뒤의 판정. 작은 글씨는 가장 압축된 지점의
모듈 폭이고, <span class="mono">1차X</span>는 보정 없이는 못 읽었다는 뜻입니다.</p>

<div class="callout">
<p><strong>문서 §8 실측의 조건은 5.5px가 아니었습니다.</strong> §8.1은 &ldquo;35° 이상 실패&rdquo;를
보고했는데, 5.5px에서는 70°까지도 보정 없이 읽힙니다. 반면
<span class="mono">d<sub>m0</sub>=2.3px</span>에서는 35°가 정확히 실패합니다 &mdash;
그리고 그때 원본 폭이 355px로, 문서가 적은 376&times;280과 거의 같습니다.</p>
<p>즉 <strong>§8의 모든 실측치는 모듈 폭 약 2.3px 조건</strong>이고,
§8.4 표의 &ldquo;35°→4.5px&rdquo;류 숫자는 5.5px를 가정해 계산한 것이라 실험 조건과 어긋나 있습니다.
§4.3의 상한 <span class="mono">m<sub>min</sub>≥0.364</span>도 같은 가정 위에 있습니다.</p>
</div>

<h2><span class="lab">정리</span>설계에 반영할 것</h2>
<ul class="find">
<li><b class="tag-fix">고칠 것</b><div><strong><code>g = cumsum(m)</code> → <code>cumsum(1/m)</code></strong>
&mdash; 해석해 대비 오차가 0.24px 대 35.1px. 디코딩으로는 안 잡히니 검증 기준에
<em>해석해 대조</em>를 넣어야 합니다.</div></li>
<li><b class="tag-fix">고칠 것</b><div><strong><code>np.clip(m, …)</code> → 기울기 전체 스케일 다운</strong>
&mdash; m만 자르면 그에 대응하는 z가 없어져 음영과 기하가 어긋납니다.</div></li>
<li><b class="tag-fix">고칠 것</b><div><strong>§15-6 &ldquo;뒤집으면 실패한다&rdquo;는 성립하지 않습니다.</strong>
20°부터 70°까지 뒤집어도 전부 읽혔습니다. 방향 검증은 디코딩이 아니라
해석해·제어점 좌표로 해야 합니다.</div></li>
<li><b class="tag-new">새로 안 것</b><div><strong>격자는 가로로 늘려야 합니다.</strong>
4&times;3의 실제 한계는 <span class="mono">λ ≥ 0.67W</span>. 6&times;3이면 0.40W까지 내려갑니다.
세로는 <span class="mono">n<sub>y</sub>=2</span>로도 충분했습니다.</div></li>
<li><b class="tag-new">새로 안 것</b><div><strong>1차 디코딩이 매우 강합니다.</strong>
5.5px에서 원통 70°까지 보정 없이 읽힙니다. 조기 종료가 자주 성공할 것이고,
Stage 2가 실제로 필요한 건 원통이 아니라 <em>짧은 파장의 주름</em>과 반사 쪽입니다.</div></li>
<li><b class="tag-new">새로 안 것</b><div><strong>판독 하한 2px는 2배 이상 보수적입니다.</strong>
합성만 놓고 보면 0.8px까지 복원됩니다. 다만 노이즈·블러·반사가 없는 조건이므로
2px는 안전 마진으로 남기되 <em>파라미터</em>로 두고 실촬영 후 조정하는 게 맞습니다.</div></li>
<li><b class="tag-ok">확인됨</b><div><strong>기울기 예산이 작동합니다.</strong>
<span class="mono">S<sub>max</sub></span>를 걸고 뽑은 표본은 원통·물결·접힘·구김을 섞어도
보정 후 전부 읽혔습니다. 성분을 <em>최대 기울기</em>라는 한 단위로 묶는 방식이 유효합니다.</div></li>
</ul>
<p class="stamp" style="margin-top:40px">
측정 조건 · zxing-cpp · OpenCV 5.0.0 · 시드 42 · 노이즈·블러·반사 없음 · 모델 없음
</p>
</div>
'''
open(f"{D}/page.html", "w").write(PAGE)
print("bytes:", len(PAGE.encode()))
