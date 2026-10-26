"""Render measured JSON into the baseline report (no model inference)."""
import json
import sys
from pathlib import Path

root=Path(__file__).resolve().parent
bench,val=map(Path,sys.argv[1:3])
b=json.loads((bench/'summary.json').read_text())
v=json.loads((val/'summary.json').read_text())
c=json.loads((bench/'run_config.json').read_text())
g=json.loads((bench/'grouped_intervals.json').read_text())
r=json.loads((bench/'retry_summary.json').read_text())
hf=json.loads((bench/'hf_provenance.json').read_text())
labels={'d0':'보정 전 D0','crop_forced_resnet':'ResNet 강제 crop',
        'crop_policy_resnet':'ResNet D0 조기 종료 crop','crop_forced_identity':'항등 강제 crop',
        'crop_policy_identity':'항등 D0 조기 종료 crop','crop_forced_gt48':'GT 48점 강제 crop',
        'crop_policy_gt48':'GT 48점 D0 조기 종료 crop',
        'oracle_resnet':'ResNet oracle','oracle_identity':'항등 oracle',
        'oracle_gt48':'GT 48점 oracle','oracle_gt4':'GT 4점 호모그래피 oracle'}
def num(value):
    return 'N/A' if value is None else f'{value:.3f}'
def pct(stat):
    return num(stat['pct'])
def ci(stat):
    return 'N/A' if stat['wilson95_pct'] is None else '–'.join(num(x) for x in stat['wilson95_pct'])
def table(summary, paths):
    lines=['| 경로 | C / W / U | EDR % | FDR % | RR_H % (C/H) | RR_H 95% Wilson | FDR_H % |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for path in paths:
        s=summary['paths'][path]; h=s['RR_H']
        lines.append(f"| {labels[path]} | {s['C']} / {s['W']} / {s['U']} | {pct(s['EDR'])} | {pct(s['FDR'])} | {pct(h)} ({h['numerator']}/{h['denominator']}) | {ci(h)} | {pct(s['FDR_H'])} |")
    return '\n'.join(lines)
res=b['paths']['crop_forced_resnet']; policy=b['paths']['crop_policy_resnet']; oracle=b['paths']['oracle_resnet']; d0=b['paths']['d0']
e=b['geometry']['metrics'];ve=v['geometry']['metrics']
timing=b['timing']
md=f'''# ResNet18-C3 기준선 재평가 — stage2-eval-v1, 2026-09-29

## 결론

기존 epoch 48 가중치를 재학습 없이 최신 평가 프로토콜로 평가했다.
고정 합성 벤치마크 **1,500장**에서 ResNet 강제 crop 경로의 정확 판독은
**{res['C']}/1500 ({pct(res['EDR'])}%)**, 오독은 **{res['W']}/1500 ({pct(res['FDR'])}%)**다.
D0가 값을 반환하면 종료하는 운영 정책을 반영한 crop 경로는
**{policy['C']}/1500 ({pct(policy['EDR'])}%)**, 오독 **{policy['W']}건**이다.
GT 출력 캔버스를 이용하는 oracle 진단은 **{oracle['C']}/1500 ({pct(oracle['EDR'])}%)**다.

이는 **기존 합성 개발 벤치마크의 기준선 측정**이다. 새 독립 최종 테스트,
실촬영 성능, Stage 1 포함 전체 E2E 성적 또는 배포 합격 판정이 아니다.
기존 validation 2,976장도 실제 split을 그대로 읽어 재평가했다.

### 결과 해석

- 항등 보정의 정답 {b['paths']['crop_forced_identity']['C']}장보다 ResNet 강제 보정은
  {res['C']-b['paths']['crop_forced_identity']['C']}장을 더 읽었다.
  같은 H에서 RR_H 개선은 {b['bootstrap']['crop_forced']['delta_RR_H_pp']:.3f}%p다.
- 운영 정책은 D0 오독 {d0['W']}건도 그대로 종료한다.
  전체 오독 {policy['W']}건 중 나머지 {policy['W']-d0['W']}건은 D0 무응답 이후의 보정 경로에서 나왔다.
  이는 실행 전이로 확인한 정책 동작이며, 모든 강제 보정 오독의 원인을 뜻하지 않는다.
- GT 48점 oracle도 {b['paths']['oracle_gt48']['W']}건 오독했다.
  과거 선별 시의 디코더 호출과 현재 native 디코더는 다르므로, 기존 target 1,500장을
  현재 디코더에서 모두 D0 실패·GT 성공인 집합이라고 가정하지 않았다.
- 이 측정에서 ResNet oracle은 GT 캔버스/단일 전체 TPS 경로, 강제 crop은 배율 재시도 경로다.
  서로 설정이 달라 oracle이라는 이름이 높은 판독률을 보장하지 않는다.

## 기준과 고정 조건

- 근거: [Stage 2 평가 프로토콜 v1](../../decisions/0005-평가-지표-체계.md).
- GitHub main 기준 commit: `{c['source_commit']}`.
- 새 평가 코드의 정확한 버전은 `run_config.json`의 파일별 SHA256으로 잠금.
- benchmark 실행 설정 SHA256: `{b['config_sha256']}`.
- validation 실행 설정 SHA256: `{v['config_sha256']}`.
- HF 모델 revision: `{hf['hf_revision']}`.
- 배포 가중치 SHA256: `{c['weights_sha256']}`.
- 데이터 revision: `{c['dataset_revision']}`.
- train / validation: 27,024 / 2,976; 학습 seed 42 단일 실행.
- checkpoint: epoch 48, **기존 val loss 최소 선택** 이력 유지. 새 지표로 재선택하지 않음.
- 모델 입력: 160×384, grayscale 3채널 반복, ImageNet 정규화. 출력 48×2.
- 원본 crop의 `(W-1, H-1)`로 좌표를 환산하고 모든 점의 2D EPE를 한꺼번에 집계.
- CPU thread=1, batch=1, FP32, TF32 off; `{c['machine']}`, `{c['gpu']}`.
- 디코더: GitHub의 `wemeet.sw.decoding.decode`를 직접 호출. 현재 pyzbar/zbar
  백엔드가 없어 **zxing-cpp {c['packages']['zxing-cpp']}** 폴백 사용.
  첫 valid/nonempty 반환값을 채택하며, Code128+원문 문자열 완전 일치만 C다.
  심볼로지와 출력은 관찰용 wrapper로 기록하고 GT는 선택 로직에 전달하지 않음.
- crop: 원본 crop 크기×1.0→1.5→3.0, 첫 반환에서 종료, TPS step=4×scale,
  INTER_CUBIC, BORDER_REPLICATE. 실제 `_rectify_and_decode` 호출.
- oracle: GT h_flat/w_flat, 전체 격자 TPS, 같은 디코더; GT를 쓰는 비배포 진단.
- `crop_policy`는 D0 반환 시 종료하는 정책을 같은 crop에서 재현한다.
  **검출기를 우회했으므로 전체 E2E가 아니다.**

## 고정 벤치마크 B — N=1,500 (low/mid/high 각 500)

D0 C/W/U = **{d0['C']}/{d0['W']}/{d0['U']}**.
H={res['RR_H']['denominator']}, H_U={policy['RR_U']['denominator']}.
EDR_B와 RR_H의 분모는 D0 재판정에 따라 위 값으로 기록한다.
실제 D0 조기 종료는 **{d0['C']+d0['W']}/1500 ({(d0['C']+d0['W'])/15:.3f}%)**이며,
여기에는 D0 오독 {d0['W']}건이 포함된다. D0 정답률과 조기 종료율은 다르다.

### 실제 crop 크기와 배율 재시도

{table(b,['d0','crop_forced_identity','crop_forced_resnet','crop_forced_gt48','crop_policy_resnet'])}

운영 보조지표 RR_U = **{policy['RR_U']['numerator']}/{policy['RR_U']['denominator']}
({pct(policy['RR_U'])}%)**, 95% Wilson **{ci(policy['RR_U'])}%**.
전체 FDR의 95% Wilson 구간은 강제 crop **{ci(res['FDR'])}%**,
D0 조기 종료 crop **{ci(policy['FDR'])}%**다.
오독을 반환한 경우도 조기 종료하며 정답을 찾을 때까지 재시도하지 않았다.

### GT 출력 캔버스 진단 — 운영 성적으로 사용 금지

{table(b,['oracle_identity','oracle_resnet','oracle_gt48','oracle_gt4'])}

## 좌표 평가 — 2D EPE

| 모집단 | 이미지 / 점 | EPE mean px | EPE P95 px | EPE RMSE px | MAE x / y px | invalid |
|---|---:|---:|---:|---:|---:|---:|
| benchmark | 1,500 / 72,000 | {num(e['epe_mean_px'])} | {num(e['epe_p95_px'])} | {num(e['epe_rmse_px'])} | {num(e['mae_x_px'])} / {num(e['mae_y_px'])} | {b['geometry']['invalid_field_count']} |
| validation | 2,976 / 142,848 | {num(ve['epe_mean_px'])} | {num(ve['epe_p95_px'])} | {num(ve['epe_rmse_px'])} | {num(ve['mae_x_px'])} / {num(ve['mae_y_px'])} | {v['geometry']['invalid_field_count']} |

단조성 위반률 benchmark={num(e['monotonicity_violation_rate']*100)}%,
validation={num(ve['monotonicity_violation_rate']*100)}%.
이 수치는 TPS 전체 영역의 무접힘 보장이 아니다.
기존 trainer의 축별/배치 평균 수치를 EPE로 바꾸어 부르지 않는다.

## Validation — N=2,976, 기존 split 그대로

{table(v,['d0','crop_forced_identity','crop_forced_resnet','crop_forced_gt48','crop_policy_resnet'])}

GT는 현재 고정된 합성 코드로 기존 레시피를 재생성했다. 재생성된 픽셀·좌표
해시도 보존했다. validation oracle와 validation 별도 지연은 미측정이며,
지연 표의 모집단은 벤치마크 1,500장이다. loss나 checkpoint를 조정하지 않았다.

## 시간과 비용

아래 각 행은 메모리의 crop에서 시작한다. 디스크 읽기·모델 로딩은 제외했다.
50회 warm-up 후 고정된 1,500장 전체 순서로 3회 측정했다.
대표값은 회차별 p50 3개의 중앙값 / p95 3개의 중앙값이다.
GPU 시간은 시작과 종료에서 동기화한다. 모델 후보를 동시에 실행하지 않았다.

| 측정 범위 | p50 ms | p95 ms | 회차별 p50 / p95 ms |
|---|---:|---:|---|
'''
for name,value in timing.items():
    if not isinstance(value,dict): continue
    passes='; '.join(f"{p['p50_ms']:.3f}/{p['p95_ms']:.3f}" for p in value['passes'])
    md+=f"| {name} | {value['representative_p50_ms']:.3f} | {value['representative_p95_ms']:.3f} | {passes} |\n"
md+=f'''
Stage 2 시간에는 전처리·장치 전송·forward·CPU 좌표/GeometryField 변환이 포함된다.
crop-policy 시간에는 D0 및 필요한 Stage 2/TPS/모든 재시도가 포함되나 Stage 1은 없다.
전체 E2E 500ms 예산 충족 여부는 이 표로 확정하지 않는다.

- 파라미터: **{c['parameters']:,}**.
- 추론 가중치: **{c['deployment_weights_bytes']:,} bytes ({c['deployment_weights_bytes']/2**20:.3f} MiB)**.
- 모델 로딩(최초 CUDA 준비 포함): {c['model_load_ms']:.3f}ms, 정상 추론 표에서 제외.
- GPU peak allocated: {timing.get('cuda_peak_allocated_bytes',0)/2**20:.3f} MiB;
  peak reserved: {timing.get('cuda_peak_reserved_bytes',0)/2**20:.3f} MiB.
  PyTorch allocator 값이며 드라이버/전체 서버 GPU 메모리가 아니다.
- 서버 이용 요금표가 없어 원화 비용은 계산하지 않았다. 이번 실행은 재학습이 아니다.

## 데이터 감사와 통계 한계

- benchmark PNG/NPZ/manifest의 발행 SHA256을 대조했다.
- 학습/검증 ID 중복 0; 평가와 학습의 정확한 레시피 중복 0(저장 필드 기준).
- 학습·검증 사이 동일 payload **{b['data_audit']['train_validation_payload_overlap']}종**.
- benchmark 고유 payload {b['data_audit']['evaluated_unique_payloads']}종 중
  **{b['data_audit']['evaluated_payloads_in_train']}종**이 학습에도 존재한다.
  같은 문자열이 곧 같은 이미지를 뜻하지는 않지만 독립된 새 송장/원본 평가를
  보장하는 분할도 아니다. canonical group_id가 없어 payload를 보수적 대리 그룹으로 썼다.
- Wilson 구간은 참고값이다. 대리 그룹 bootstrap(10,000회, seed42)도
  `grouped_intervals.json`에 EDR/FDR/RR_H/FDR_H/RR_U별로 보관했다.
  대리 그룹은 실제 원본 독립성을 증명하지 않는다.
- 오독 관측 0건의 bootstrap percentile 구간이 [0,0]이어도 위험 0의 증명이 아니다.
- 단일 학습 seed 결과이며, 새로운 holdout/실촬영 데이터의 최종 확인이 남아 있다.
- validation과 benchmark를 합산한 판독률은 계산하지 않았다.

## 다음 단계

1. 오독 ID와 D0→최종 반환 전이를 검토한다. 구제율 상승만으로 배포를 승인하지 않는다.
2. 이 고정 기준선과 동일 split/경로에서 학습률 또는 loss beta를 한 번에 하나씩 비교한다.
3. 그룹 분리된 새로운 holdout과 실촬영 세트를 확보하고 설정 잠금 후 최종 평가한다.
4. Stage 1 검출기를 포함해 원본 사진에서 전체 E2E 성능·지연을 측정한다.

## 재현과 산출물

```bash
bash docs/experiments/2026-09-29-stage2-eval-v1/run.sh
```

`BARCODE_DATA_ROOT`, `BARCODE_PYTHON`, `CUDA_VISIBLE_DEVICES`, `BARCODE_EVAL_OUTPUT`로
로컬 데이터·환경·GPU·출력 위치를 지정할 수 있다. 기존 결과 폴더를 덮어쓰지 않는다.
W&B는 사용자 요청에 따라 사용하지 않았다.

- [상세 측정표](RESULTS.md): bucket/band/preset, 재시도, 전이표, 신뢰구간.
- [집계 JSON](results.json): 원시 latency 배열을 제외한 요약, hash, 데이터 감사.
- 로컬 원자료: `runs/stage2-eval-v1-benchmark-20260929/`,
  `runs/stage2-eval-v1-validation-20260929/`.
- 각 폴더에 run_config, per_sample, sample_manifest, input_hashes, hard_ids,
  summary, grouped_intervals, retry_summary, outcomes.csv, artifact_hashes 저장.
- 벤치마크에 timing.json: 모든 3회 원시 지연 배열. per_sample의 관찰용 시간은
  진단값이며 공식 지연 통계와 섞지 않는다.
'''
(root/'README.md').write_text(md)
full=['# 측정표 전문\n']
for title,summary,directory in [('benchmark (1500)',b,bench),('validation (2976)',v,val)]:
    full+=[f'## {title}',f"설정 SHA256: `{summary['config_sha256']}`",table(summary,list(summary['paths']))]
    for key in ['bucket','band','preset']:
        full.append(f'### {key}별 강제 crop ResNet')
        full+=['| 구분 | N | C/W/U | EDR % | FDR % | RR_H % | EPE mean/P95 px |',
               '|---|---:|---:|---:|---:|---:|---:|']
        for name,item in summary['by_'+key].items():
            s=item['paths']['crop_forced_resnet']; m=item['geometry']['metrics']
            full.append(f"| {name} | {s['n']} | {s['C']}/{s['W']}/{s['U']} | {pct(s['EDR'])} | {pct(s['FDR'])} | {pct(s['RR_H'])} | {num(m['epe_mean_px'])}/{num(m['epe_p95_px'])} |")
    full.append('### D0 조기 종료 crop ResNet 전이표')
    full+=['| D0 → 최종 | 장수 |','|---|---:|']
    for label,count in summary['paths']['crop_policy_resnet']['transitions'].items():
        full.append(f'| {label} | {count} |')
    extra=json.loads((directory/'retry_summary.json').read_text())
    full.append('### 강제 crop ResNet 배율별 실제 누적 성적')
    full+=['| 배율 | 실제 실행 N | C/W/U | 추가 C | 추가 W | 추가 처리시간 평균 ms (전체 N 기준, 진단) |','|---|---:|---:|---:|---:|---:|']
    for stage in extra['crop_forced_resnet']:
        t=stage['cumulative']
        full.append(f"| {stage['scale']} | {stage['called_n']} | {t['C']}/{t['W']}/{t['U']} | {stage['added_C']} | {stage['added_W']} | {num(stage['added_time_ms_mean_all_samples'])} |")
    full.append('### 항등 대비 paired ΔRR_H, %p')
    full.append('대리 payload group bootstrap 10,000회, seed42; 실제 원본 독립성 미확인.')
    for path,s in summary['bootstrap'].items():
        full.append(f"- {path}: {num(s['delta_RR_H_pp'])}%p, 95% CI {s['ci95_pp']}, 유효 {s['valid_draws']}/{s['draws']}.")
    grouped=json.loads((directory/'grouped_intervals.json').read_text())
    full+=['### ResNet 신뢰구간 (95%, %)','| 경로/지표 | 분자/분모 | % | Wilson (참고) | payload group bootstrap |','|---|---:|---:|---|---|']
    for path in ['crop_forced_resnet','crop_policy_resnet']:
        for metric in ['EDR','FDR','RR_H','FDR_H','RR_U']:
            s=summary['paths'][path][metric]
            if s is None: continue
            interval=grouped['paths'][path][metric]['ci95_pct']
            full.append(f"| {path}/{metric} | {s['numerator']}/{s['denominator']} | {pct(s)} | {ci(s)} | {interval} |")
(root/'RESULTS.md').write_text(('\n\n'.join(full)+'\n').replace('|\n\n|', '|\n|'))
# Do not duplicate 3xN raw timing arrays in Git; preserve them in the run directory.
compact=json.loads(json.dumps(b))
for value in compact.get('timing',{}).values():
    if isinstance(value,dict):
        for p in value['passes']: p.pop('raw_ms',None)
(root/'results.json').write_text(json.dumps({'benchmark':compact,'validation':v,'config':c,'hf':hf},ensure_ascii=False,indent=2)+'\n')
print(root/'README.md')
