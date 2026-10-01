# LTN Gradient Conflict Study

## 목적
LTN(LTNtorch)에서 neuro 손실(숫자 라벨 CE)과 symbolic 손실(1 - SatAgg(공리))의 기울기 충돌을
측정하고, 충돌 해결 기법이 성능·효율·강건성을 개선하는지 검증한다.

## 원칙
- 업데이트 방식은 src/aggregators/의 GradAggregator 인터페이스로만 바꾼다. 기본값은 sum (L_neural + lam * L_sym).
- 기울기는 torch.autograd.grad(..., retain_graph=True)로 공유 backbone 파라미터에 대해 따로 계산한다.
- 충돌은 두 수준에서 잰다.
  (1) 기울기 수준: cos(g_n, g_s)
  (2) 업데이트 수준: cos(Δθ, g_n), cos(Δθ, g_s). Δθ는 optimizer.step() 전후의 파라미터 차이.
- LTNtorch와 외부 코드의 API는 추측하지 않는다. 설치된 소스를 먼저 읽는다.
- 평가 지표(성능·효율·안정성)는 실험 전에 확정된 목록만 쓴다. 결과를 보고 지표를 추가하지 않는다.
  꼭 필요하면 "사후 추가"라고 명시한다.
- 모든 방법에 같은 튜닝 예산, 같은 seed, 같은 epoch를 쓴다. 선택은 validation 기준.
- 재현성: config(YAML) + seed. 결과는 runs/<exp>/<seed>/ 에 metrics.jsonl, config.yaml로 저장.
- 각 Phase가 끝나면 멈추고 보고한다. 실행 전에는 실행 수와 예상 시간(1 epoch 실측 기반)을 알려주고 승인받는다.
- 통계적으로 유의하지 않은 차이는 그렇다고 명시한다. 결과를 과장하지 않는다.

## 구조
src/data/ (mnist_addition.py), src/kb/, src/models/, src/aggregators/, src/grad_probe.py, src/train.py
third_party/ (DeepSoftLog, A-NeSI 공식 코드. 수정 사항은 patches/에 patch 파일로)
configs/, scripts/ (sweep.py, plot.py), tests/, RESULTS.md

## 명령
- 설치: pip install -r requirements.txt
- 테스트: pytest -q
- 실행: python -m src.train --config configs/<name>.yaml --seed 0

## 실험환경
- conda를 통한 본 실험 전용 가상환경 생성
- 학습 시간이 오래걸릴거 같으면 colab mcp 사용