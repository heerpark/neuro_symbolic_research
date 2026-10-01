# 진행 상황 (이어서 작업하기 위한 메모)

마지막 갱신: Phase 0 완료 직후. **Phase 1은 아직 시작 안 함 — 사용자 승인 대기 중.**

## 지금 상태 한 줄 요약
Phase 0(기준 성능 재현) 통과. DeepSoftLog·A-NeSI 둘 다 ±1%p 이내 재현 성공 → Phase 2에서 둘 다 사용. 다음 작업은 Phase 1(충돌 측정) 구현인데, 아직 코드를 쓰지 않았다.

## 읽어야 할 파일 (우선순위 순)
1. `CLAUDE.md` — 전체 연구 스펙 (Phase 0/1/2 정의, 원칙, 명령어). 모든 작업의 근거.
2. `RESULTS.md` — Phase 0 표와 결론. Phase 1/2는 아직 틀만 있음.
3. `third_party/README.md` — DeepSoftLog/A-NeSI 재현 중 발견한 업스트림 버그, 적용한 수정, pinned commit, conda env 구성법. **재실행하거나 third_party를 다시 clone할 때 반드시 참고.**
4. `/Users/phr/.claude/plans/claude-md-keen-peacock.md` — Phase 0 구현 계획(이미 실행 완료, 참고용).

## 환경 (이미 구축 완료, 재사용만 하면 됨)
- `neuro_symbolic` conda env: 우리 LTN 코드(`src/`), 테스트, 결과 집계 스크립트 전용. python 3.10.21, torch 2.14.0, LTNtorch 1.0.2, torchvision 0.29.0, pyyaml, pytest 설치됨.
- `dsl_env` conda env: DeepSoftLog 전용. python 3.10. **pysdd는 requirements.txt의 0.2.11이 아니라 1.0.6으로 설치해야 함**(0.2.11의 macOS arm64 prebuilt 바이너리가 깨져 있음 — 근거는 third_party/README.md). `setuptools<81`도 필요(wandb의 pkg_resources 의존성 때문).
- `anesi_env` conda env: A-NeSI 전용. python 3.9.15. `third_party/patches/anesi/environment.osx-arm64.yml`로 생성(mkl 제거판). `pip install -e .` 이후 **반드시** `pip install --no-deps torch==1.12.1 torchvision==0.13.1`로 재고정해야 함(그냥 두면 pip가 최신 torch로 올려버림).
- third_party 코드 자체 수정: `third_party/patches/anesi/0001-fix-train_all-TrainResult-unpacking.patch` 한 건 적용됨(실제 clone에도 이미 반영돼 있음). DeepSoftLog는 소스 수정 없음.

## 지금 당장 재실행하고 싶으면
```bash
# 테스트
conda run -n neuro_symbolic python -m pytest -q tests/

# Phase 0 전체 재실행 (이미 끝났지만 재현 확인용)
conda run -n neuro_symbolic python scripts/sweep.py --parallel 4

# 결과표 재생성
conda run -n neuro_symbolic python scripts/make_results_table.py
```
`runs/<exp>/<seed>/` 아래 모든 결과가 남아있음 (`runs/_pilot/`은 파일럿 타이밍 측정용, 무시해도 됨).

## Phase 0 결과 요약
| 방법 | N | 논문 | 재현 | 통과 |
|---|---|---|---|---|
| LTN | 1 | 80.54±23.33 (gate 없음) | 96.27±0.75 | 참고용 |
| LTN | 2 | 77.54±35.55 (gate 없음) | 94.12±0.13 | 참고용 |
| DeepSoftLog | 1 | 98.40±0.10 | 98.16±0.07 | **PASS** |
| DeepSoftLog | 2 | 96.60±0.30 | 95.91±0.16 | **PASS** |
| A-NeSI (predict) | 1 | 97.66±0.21 | 97.79±0.29 | **PASS** |
| A-NeSI (predict) | 2 | 95.96±0.38 | 96.14±0.20 | **PASS** |

## 다음 할 일: Phase 1 (아직 코드 없음)
CLAUDE.md의 Phase 1 스펙(조건: N∈{1,2} × rule_noise∈{0,0.2} × seed 3, probe_every=50, Gate 1)을 구현해야 하는데, 이를 위해 Phase 0에서 아직 안 만든 것들:

1. **"ours" 데이터 설정**: 현재 `src/data/mnist_addition.py`는 "standard"(합 라벨만)만 지원. `labeled_frac`(기본 0.01, 숫자 라벨 일부 공개)과 `rule_noise`(합 라벨 일부를 틀린 값으로 교체)를 추가해야 함.
2. **`src/grad_probe.py` 실제 구현**: 지금은 `get_shared_backbone_params()` 스텁만 있음. 실제로 필요한 것:
   - `torch.autograd.grad(L_neural, shared_params, retain_graph=True)`와 `torch.autograd.grad(L_sym, shared_params, retain_graph=True)`를 따로 계산
   - 기울기 수준: cos(g_n, g_s) (전체 + 레이어별), ‖g_s‖/‖g_n‖, 누적 충돌 비율(cos<0)
   - 업데이트 수준: optimizer.step() 전후 Δθ 계산, cos(Δθ, g_n), cos(Δθ, g_s)
3. **neuro 손실(L_neural) 추가**: Phase 0 standard 설정엔 없었음(순수 symbolic). `DigitCNN.logits()`는 이미 만들어뒀으니 CE 손실만 연결하면 됨.
4. **tests/**: "같은 손실 두 개 → cos=1, 부호만 반대 → cos=-1" 테스트 추가.
5. **configs/**: N∈{1,2} × rule_noise∈{0,0.2} 조합 4개 config 작성 (seed는 CLI로 받음, 기존 패턴과 동일).
6. 진행 순서는 CLAUDE.md 원칙대로: 구현 → 테스트 → 작은 시험 실행(seed 1, 조건 1) → 실행 수·예상 시간 보고 → 승인 → 전체 실행 → Gate 1 결과 보고 → 정지.

**다음 세션에서 바로 시작할 수 있는 첫 질문**: "Phase 1 구현 계획을 먼저 보여줘" 또는 바로 구현 시작 — 사용자가 Phase 1 진행을 승인했는지부터 확인할 것.
