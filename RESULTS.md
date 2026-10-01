# Results

## Phase 0: 기준 성능 재현

전체 18회 실행(방법 3 × N∈{1,2} × seed 3) 완료. 실행 환경·재현 절차·발견한 업스트림 버그와 수정 내역은 `third_party/README.md` 참고.

<!-- PHASE0_TABLE_START -->
| 방법 | N | 논문 수치 | 재현 수치 | 통과 여부 |
|---|---|---|---|---|
| LTN (LTNtorch, standard) | 1 | 80.54 ± 23.33 | 96.27 ± 0.75 | — (gate 없음, 참고용) |
| LTN (LTNtorch, standard) | 2 | 77.54 ± 35.55 | 94.12 ± 0.13 | — (gate 없음, 참고용) |
| DeepSoftLog (공식 코드) | 1 | 98.40 ± 0.10 | 98.16 ± 0.07 | PASS |
| DeepSoftLog (공식 코드) | 2 | 96.60 ± 0.30 | 95.91 ± 0.16 | PASS |
| A-NeSI (predict, 공식 코드) | 1 | 97.66 ± 0.21 | 97.79 ± 0.29 | PASS |
| A-NeSI (predict, 공식 코드) | 2 | 95.96 ± 0.38 | 96.14 ± 0.20 | PASS |
<!-- PHASE0_TABLE_END -->

**판정: 두 SOTA 기준 모두 ±1%p 이내로 통과.** 스펙 규칙에 따라 DeepSoftLog와 A-NeSI 둘 다 Phase 2의 7번(SOTA + 숫자 라벨 CE) 비교 대상으로 사용한다.

**LTN 자체 재현 (참고, gate 없음):** N=1 96.27±0.75, N=2 94.12±0.13으로, A-NeSI 논문 Table 1이 보고한 원본 LTN 수치(N=1 80.54±23.33, N=2 77.54±35.55)보다 평균이 훨씬 높고 seed 간 표준편차도 극적으로 작다(예: N=1 표준편차 23.33 → 0.75). 가능한 원인: (1) LTNtorch의 `stable=True` 연산(AggregPMean/PMeanError의 π₀/π₁ 안정화)이 원 TensorFlow 구현보다 수치적으로 안정적일 수 있음, (2) 본 재현은 Exists의 p를 1→2→4→6으로 anneal했고 Forall의 p는 2로 고정했는데, 이 스케줄이 우연히 이 벤치마크에 잘 맞았을 수 있음, (3) 원 LTN 수치의 출처가 A-NeSI 논문의 재구현이라 우리와 하이퍼파라미터·백본 CNN 구조가 다를 수 있음. 이 차이의 정확한 원인 규명은 본 연구의 범위를 벗어나므로 사후 분석으로 명시하고 넘어간다.

**LTN-SOTA 격차:** 우리 LTN 재현은 두 SOTA보다 여전히 낮다 — DeepSoftLog 대비 N=1 −1.89%p, N=2 −1.79%p; A-NeSI 대비 N=1 −1.52%p, N=2 −2.02%p. Phase 1/2에서 측정할 기울기 충돌이 이 격차와 관련 있는지가 핵심 질문 중 하나가 된다.

**Phase 0 결론: 통과. Phase 1로 진행 가능.** (본 보고 후 정지 — 사용자 승인 후 Phase 1 시작)

## Phase 1: 충돌 측정

(Phase 0 통과 후 진행)

## Phase 2: 비교

(Phase 1 통과 후 진행)
