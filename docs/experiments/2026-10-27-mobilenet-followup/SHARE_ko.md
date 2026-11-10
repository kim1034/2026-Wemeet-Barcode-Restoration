# Notion 및 GitHub 공유 방법

## 준비한 내용

- `DECISION_ko.md`: 두 운영 검증 후보와 핵심 결과.
- `NOTION_ko.md`: 기존 네 조건 비교와 이번 후속 실험을 합친 문서.
- `REPORT_ko.md`, CSV/JSON, PNG/SVG: 전체 결과, 실패 분석, 그래프 및 검증 근거.
- `PR_BODY.md`: GitHub PR 본문.
- 이전 실험 폴더: 기존 3-seed 비교, paired 구간 12개, 당시 시간 측정.

Notion 원격 페이지와 GitHub 원격 브랜치는 아직 게시하지 않았다.
2026-10-27에 확인한 연결 계정 `yeonwoo29`의 대상 저장소 권한은 `pull=true`, `push=false`다.
쓰기 권한이 있는 계정에서 게시할 수 있도록 로컬 커밋과 공유 파일을 만든다.

## Notion

1. 제공된 `notion-ready.zip`을 푼다.
2. `2026-10-27-mobilenet-followup/NOTION_ko.md`를 가져오거나 본문을 붙여 넣는다.
3. 같은 폴더의 그래프 세 개를 본문 이미지 위치에 첨부한다. CSV/JSON은 필요에 따라 첨부한다.
4. 간단한 결론만 필요하면 `DECISION_ko.md`를 사용한다. 두 후보는 모두 유지한다.

이미지와 다른 문서의 상대 경로가 자동 연결되지 않으면 압축파일에 포함된 파일을 첨부한다.

## GitHub

저장소: https://github.com/kim1034/2026-Wemeet-Barcode-Restoration

로컬 브랜치: `ai/mobilenet-followup-20261027`.
공유 패키지의 `manifest.json`에 실제 최종 commit SHA와 파일 hash가 들어 있다.
가중치·원본 이미지·인증 정보는 커밋하지 않는다. 원시 기록은 W&B artifact로 연결한다.

현재 서버의 완성된 작업 디렉터리에서는 쓰기 권한이 있는 인증을 사용해 다음 브랜치를 push한다.

```bash
git push -u origin ai/mobilenet-followup-20261027
```

다른 컴퓨터의 새 clone에서는 제공된 bundle을 가져올 수 있다. 경로는 실제 다운로드 위치로 바꾼다.

```bash
git fetch /path/to/github-branch.bundle ai/mobilenet-followup-20261027:ai/mobilenet-followup-20261027
git switch ai/mobilenet-followup-20261027
git push -u origin ai/mobilenet-followup-20261027
```

bundle 대신 patch를 쓸 경우, 이 실험의 기반 commit에서 새 브랜치를 만든다.

```bash
git switch -c ai/mobilenet-followup-20261027 f449b9c34df3a42c5c324a0629b0dbee3308f1b3
git am /path/to/github-changes.patch
git push -u origin ai/mobilenet-followup-20261027
```

이후 `main`을 대상으로 PR을 만들고 `PR_BODY.md`를 본문으로 사용한다.
저장소의 CONTRIBUTING 규칙에 따라 같은 파트 리뷰어를 지정하고 승인 후 merge한다.
이 패키지를 적용하는 것만으로 제품의 기본 판독 정책이 바뀌지는 않는다.
측정이 완료된 GPU service를 다시 실행할 필요도 없다.
