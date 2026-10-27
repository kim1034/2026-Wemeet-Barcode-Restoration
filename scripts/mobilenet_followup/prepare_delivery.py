"""Prepare reviewed Notion/GitHub text after both completion gates pass."""

import argparse
import re
import shutil
from pathlib import Path

from scripts.controlled_geometry.common import read_json, write_json
from scripts.mobilenet_followup.core import POLICIES
from scripts.mobilenet_followup.report import LABELS, table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--priority", choices=["misread", "accuracy", "both"], default="both")
    args = parser.parse_args()
    root = args.root.resolve()
    done = read_json(root / "completion.json")
    audit = read_json(root / "delivery/completion.json")
    assert done["status"] == audit["status"] == "complete"
    repo = Path(__file__).resolve().parents[2]
    docs = repo / "docs/experiments/2026-10-27-mobilenet-followup"
    summaries = read_json(root / "report/summary.json")
    candidates = []
    for key, summary in summaries["accuracy"].items():
        if not key.startswith("scene-mobilenetv3-small-"):
            continue
        for policy in POLICIES:
            candidates.append(
                {
                    "task": key,
                    "policy": policy,
                    **summary["paths"][policy],
                    **summaries["timing"][key + "-timing"][policy],
                }
            )
    conservative = min(candidates, key=lambda r: (r["W"], -r["C"], r["p95_ms"]))
    maximum = min(candidates, key=lambda r: (-r["C"], r["W"], r["p95_ms"]))

    def description(r):
        size, mode = r["task"].split("-")[-2:]
        return f"imgsz {size} · CLAHE {'on' if mode == 'clahe' else 'off'} · {LABELS[r['policy']]}"

    selected = maximum if args.priority == "accuracy" else conservative
    focus = (
        "아래 두 후보를 별도의 운영 검증 조건으로 유지한다."
        if args.priority == "both"
        else f"우선 검증할 설정: **{description(selected)}**."
    )
    title = (
        "운영 후보 두 가지를 유지한다."
        if args.priority == "both"
        else "오독 억제를 우선한 운영 검증 후보를 권한다."
        if args.priority == "misread"
        else "정답률을 우선한 운영 검증 후보를 권한다."
    )
    early = summaries["accuracy"]["crop-benchmark-clahe"]["paths"]["early_exit"]
    forced = summaries["accuracy"]["crop-benchmark-clahe"]["paths"]["always_rectify"]
    agreement = summaries["accuracy"]["crop-benchmark-clahe"]["paths"]["agreement"]
    native = read_json(root / "delivery/native_pipeline_audit.json")
    validation = read_json(root / "delivery/independent_validation.json")
    tests = (root / "delivery/check-4.log").read_text()
    match = re.search(r"(\d+) passed, (\d+) skipped", tests)
    assert match
    passed, skipped = match.groups()
    decision = [
        "# 최종 판단과 공유용 요약",
        "",
        "백본은 **MobileNet M, seed42, epoch90**을 유지한다. " + title,
        "M: lr 3e-4, batch 64, 100 epoch 학습, β=1/154. 검증 기준 선택은 epoch90이다.",
        "새 학습 없이 기존 선택 체크포인트로 추가 실험했다.",
        "",
        focus,
        "",
        "아래는 합성 장면 100개에서 관찰한 결과다. 오독률은 전체 입력 대비 비율이다.",
        "",
        table(
            ["후보", "설정", "정답", "오독", "무응답", "p50 / p95(ms)"],
            [
                [
                    name,
                    description(r),
                    r["C"],
                    r["W"],
                    r["U"],
                    f"{r['p50_ms']:.2f} / {r['p95_ms']:.2f}",
                ]
                for name, r in [("오독 억제 우선", conservative), ("정답률 우선", maximum)]
            ],
        ),
        "",
        "정책·해상도는 이 개발 결과에서 고른 후보이며, 독립 test로 검증된 배포 설정은 아니다.",
        "특히 100개에서 오독 0개를 관찰해도 실제 오독 위험이 0이라고 판단할 수 없다.",
        "",
        "## crop에서 확인한 정책 효과",
        "",
        f"CLAHE benchmark 1,500개에서 항상 보정은 현재 정책 대비 정답률 "
        f"{early['EDR']:.2f}%→{forced['EDR']:.2f}%, 오독률 "
        f"{early['FDR']:.2f}%→{forced['FDR']:.2f}%였다.",
        f"일치 확인은 오독 {agreement['W']}/1,500개로 줄었지만 "
        f"무응답 {agreement['U']}개({agreement['NRR']:.2f}%)를 만들었다.",
        "crop 결과와 전체 장면 결과를 구분했다. 탐지 crop에서는 강제 보정의 손실도 있었다.",
        "기존 네 조건은 3개 seed의 평균±표준편차, 후속 실험은 단일 체크포인트 결과다.",
        "MobileNet은 공통 head를 적용한 새 후보다. 과거 보고서 모델의 정확한 재현은 아니다.",
        "CLAHE와 백본 효과를 분리한다. MobileNet이 항상 더 빠르다고 결론내리지 않는다.",
        "",
        "## 완료 및 검증",
        "",
        f"정확도 {done['accuracy_tasks']}개, 시간 측정 {done['timing_tasks']}개가 모두 완료됐다.",
        "각 시간 작업은 50회 warmup 후 세 정책을 전체 입력에서 각각 3회 반복했다.",
        "측정에는 판독 추적·이미지 hash를 포함한다. 모델 로딩·파일 I/O·UI는 제외한다.",
        "항상 보정은 D0 판독을 생략한다. 일치 확인은 D0와 세 warp를 모두 실행한다.",
        f"CPU 대조 {len(native['tasks'])}개 장면 조건×100개 판독이 일치했다. "
        "기록된 detector/model 출력을 사용한 검사이며 GPU 재추론으로 표현하지 않는다.",
        f"W&B 평가/시간 작업 22개의 finished 상태와 artifact 파일 digest를 확인했다. "
        f"원본·코드·수정 기록 보존 검사 {validation['preservation_checks']}건이 통과했다.",
        f"최종 코드 검사: pytest {passed} passed, {skipped} skipped; "
        "ruff check/format 및 import-linter 4개 계약 통과.",
        "실행 중 native 대조 설정 수정과 실패 로그는 EXECUTION_ko.md에 공개했다.",
        "",
        f"[통합 W&B 결과]({done['wandb']}) · [독립 최종 검증]({audit['wandb']})",
        "",
        "## 공유 파일",
        "",
        "Notion에는 NOTION_ko.md를 사용한다. 전체 수치·실패 샘플·재현 설명은 같은 폴더에 있다.",
        "GitHub 브랜치는 ai/mobilenet-followup-20261027이다. 원격 게시 여부는 별도 확인한다.",
        "가중치와 데이터는 GitHub에 넣지 않고 고정된 HF revision 및 W&B artifact로 연결한다.",
        "",
    ]
    (docs / "DECISION_ko.md").write_text("\n".join(decision))
    base_report = (root / "report/REPORT_ko.md").read_text()
    prior_latency = (
        repo / "docs/experiments/2026-10-26-stage2-controlled-v2/LATENCY_ko.md"
    ).read_text()
    notion = (
        "\n".join(decision)
        + "\n---\n\n"
        + base_report
        + "\n---\n\n"
        + prior_latency
        + "\n## 그래프\n\n![기존 네 조건](prior_four_way.png)\n\n"
        "![crop 정책 비교](crop_policy_tradeoffs.png)\n\n"
        "![장면 정확도와 시간](scene_accuracy_latency.png)\n"
    )
    (docs / "NOTION_ko.md").write_text(notion)
    for name in (
        "native_pipeline_audit.json",
        "independent_validation.json",
        "controlled_inputs_match.json",
    ):
        shutil.copy2(root / "delivery" / name, docs / name)
    body = (
        "기존 ResNet/MobileNet의 서로 다른 학습 조건을 공통 head와 동일 데이터에서 "
        "R/M 두 preset, seed42/43/44로 교차 비교한 결과를 기록합니다. "
        "선택된 MobileNet M에 대해 판독 정책, 전체 장면 경로, "
        "탐지 해상도, 실패 분석을 추가했습니다.\n\n"
        f"오독 억제 우선 후보는 {description(conservative)}이며, 합성 장면 100개에서 "
        f"정답 {conservative['C']}개/오독 {conservative['W']}개, "
        f"평가 경로 p95 {conservative['p95_ms']:.2f}ms입니다. "
        f"정답률 우선 후보는 {description(maximum)}, "
        f"정답 {maximum['C']}개/오독 {maximum['W']}개, p95 {maximum['p95_ms']:.2f}ms입니다. "
        "두 설정을 모두 운영 검증 후보로 유지합니다. "
        "추가 학습이나 제품 기본 정책 변경은 포함하지 않습니다.\n\n"
        "- 정확도 12개 및 직렬 시간 측정 10개, W&B 저장 파일 대조 완료.\n"
        "- 원시 판독/분모/집계/hash 검사, 장면 800건 native 파이프라인 대조 완료.\n"
        f"- pytest {passed} passed/{skipped} skipped, ruff 및 import-linter 통과.\n"
        "- 합성 개발셋에서 선택한 운영 후보이며 별도 실촬영 holdout 검증이 필요합니다.\n\n"
        "읽을 문서: `docs/experiments/2026-10-27-mobilenet-followup/DECISION_ko.md`, "
        "`NOTION_ko.md`, `REPORT_ko.md`, `EXECUTION_ko.md`.\n"
    )
    (docs / "PR_BODY.md").write_text(body)
    write_json(
        root / "delivery/recommendation.json",
        {
            "priority": args.priority,
            "conservative": conservative,
            "maximum_accuracy": maximum,
            "selected": None if args.priority == "both" else selected,
            "selection_population": "synthetic development scenes, n=100",
        },
    )
    print(
        {
            "priority": args.priority,
            "candidates": [description(conservative), description(maximum)],
            "selected": None if args.priority == "both" else description(selected),
            "tests_passed": passed,
        }
    )


if __name__ == "__main__":
    main()
