# VINN Real-World RL Fine-tuning Plan (retrieval re-ranking bandit)

작성: 2026-08-06. 전제: 필터 뱅크(`left_infonce_mask12_filt3e3.npz`, 72,643 frames) + mask12_e6
인코더 + Episode Manager 스택. 핵심 아이디어: **정책의 결정을 "top-k 후보 중 1개 선택"이라는
이산 문제로 바꾸고, 뱅크 엔트리별 가치를 실기 rollout의 성공/실패로 학습**한다 (Neural Episodic
Control식 — 뱅크 = Q 테이블). 액션은 항상 데모된 chunk이므로 탐색이 convex-hull 안에서 안전하고,
Ruckig이 최종 방어선.

## 0. 문제 정식화
- **State**: 쿼리 임베딩 e (768-d) + phase 신호(직전 선택 엔트리의 (epi, frame), 경과 tick).
- **Action**: 거리 기준 top-k(=32) 후보 중 엔트리 1개 선택 → 그 엔트리의 chunk를 H tick 실행
  (macro-action, H=8 기본) 후 재쿼리. 30s 에피소드 ≈ 110~120 결정.
- **Reward**: 에피소드 종료 시 R∈{0,1} (성공/실패) − λ·(소요시간). FT 임계 초과/E-STOP은 R=0 처리.
- **Episode 경계**: Episode Manager의 START/STOP 그대로 사용.

## 1. 선택 규칙 (serving)
```
score_i = exp(-d_i / bw) × v_i × prog_i
```
- `v_i`: 엔트리 가치. Thompson 샘플 `v_i ~ Beta(α_i, β_i)` (학습 전 α0=3, β0=1 — 데모=성공 prior).
  Thompson이 곧 탐색: 근소한 후보끼리만 자연스럽게 섞이고, 명백히 나쁜 엔트리는 뽑히지 않음.
- `prog_i`: 진행 prior — 직전 선택과 같은 에피소드면 frame이 앞서는 후보를 선호 (γ^max(0, f_prev−f_i)).
  loop-break 휴리스틱의 일반화이며 aliasing(부분관측) 완화 장치. 필터 뱅크로 이미 절반은 해결됨.
- argmax 1개 실행. temporal ensemble은 유지 (겹치는 top-1 chunk 8개 평균 → 평활 회복).
- **Fallback**: top-k 전부 v가 낮으면(모든 후보 실패 이력) 기존 softmin 가중평균으로 후퇴 + 로그 플래그.

## 2. 학습 규칙 (에피소드 사이, 오프라인 업데이트 — 30Hz 루프 안에서는 학습 없음)
- **비대칭 크레딧**: 성공 질량 R은 선택된 전체 경로에 균등 (`α_i += R`) — 성공은 경로 전체의
  검증. 실패 질량 (1−R)은 **로그 끝(=실패 시점)에서 뒤로 지수 감쇠**
  (`β_i += (1−R)·0.5^(Δt/반감기 3s)`) — 실패 직전 선택이 벌점을 지고 초반 접근은 면책.
  전체-에피소드 균등 벌점의 credit 희석 문제(결정 ~110개가 공동 책임)를 직접 해결.
- **운영 규칙: 실패라고 판단되는 즉시 STOP** — 로그의 마지막 틱이 실패 타임스탬프가 되므로
  별도 라벨 포맷 없이 실패 시점이 기록됨. (E-STOP 반응 지연 ~1 s는 반감기 3 s가 흡수.)
- **시간 전파**: 선택 엔트리 i의 같은-에피소드 이웃 j (|f_j − f_i| ≤ W=15)에도 감쇠 c=0.3으로
  같은 업데이트 (같은 데모 구간은 운명 공동체 — 테이블 평활 + 학습 가속).
- 가치 테이블은 뱅크 행에 정렬된 사이드카 npz (`values_left_filt3e3.npz`: alpha, beta).
  업데이트 스크립트가 에피소드 로그를 소비해 재작성 → 서버는 reset 시 리로드.

## 3. 구현 항목
| # | 작업 | 상태 |
|---|---|---|
| 1 | `knn_runner`: `VPI_KNN_SELECT={softmin,top1,thompson}` + prog prior + 가치 사이드카 로드 | **DONE** (2026-08-06, 스모크 통과: 선택/가치반영/fallback/로그회전) |
| 2 | tick 로그: (t, top-k idx/dist, 선택 idx, fallback 여부) → 에피소드별 JSONL (`VPI_KNN_LOG_DIR`) | **DONE** |
| 3 | frontend Episode Control에 **SUCCESS / FAIL 버튼 2개** (수동 라벨 v1) — 그때까지는 `labels.csv` 수기 | TODO |
| 4 | `update_values.py`: 로그+라벨 → Beta 업데이트(시간 전파 포함) → 사이드카 npz | **DONE** (합성 데이터로 수치 검증) |
| 5 | (v2) hook_pose_estimator 서비스로 에피소드 종료 시 자동 성공 판정 | TODO |

실기 기동 예시:
```bash
export VPI_KNN_BANK=left_vinn/banks/left_infonce_mask12_filt3e3.npz
export VPI_KNN_SELECT=thompson VPI_KNN_LOG_DIR=rl_logs VPI_KNN_VALUES=values_left_filt3e3.npz
# 에피소드 라벨: rl_logs/labels.csv 에 "<ep 파일명>,<1|0>" 추가 후
python update_values.py --logs rl_logs --labels rl_logs/labels.csv \
  --bank $VPI_KNN_BANK --out values_left_filt3e3.npz   # 서버 재시작(또는 reset) 시 반영
```

## 3b. 에피소드 채점 루브릭 (labels.csv 점수 기준)
점수는 [0,1] 실수. **처음 30 ep은 1.0/0.0 이분법만 사용** (라벨 일관성 확보가 우선), 부분
점수는 그 후에. Beta 업데이트가 소수를 그대로 소화하므로 포맷 변경 없음 (α+=R, β+=1−R).

| 점수 | 기준 |
|---|---|
| **1.0** | hook이 목표 행어에 걸리고, 손을 뗀 뒤 **3초 이상 유지**, 기준 시간 내 완료 |
| 0.7 | 성공했으나 기준 시간 초과, 또는 흔들림/재시도 끝에 성공 |
| 0.3 | 행어에 닿았으나 미체결/이탈 (거의 성공 — 접근·정렬은 유효했다는 신호) |
| **0.0** | 미도달, 낙하, 타임아웃, FT 임계 초과, 충돌, E-STOP, **오퍼레이터 개입으로 성공** |
| (제외) | 정책과 무관한 외부 요인 중단 (케이블, 외부 접촉 등) — labels.csv에 **줄을 쓰지 않음** |

- 기준 시간: baseline 20–30 ep의 완료시간 p75로 확정 (그 전 임시값 45 s).
- 시간 페널티는 별도 λ항 대신 위 1.0/0.7 구분으로 흡수 (Beta의 Bernoulli 의미 유지).
- 판정 근거 우선순위: ① 육안 (v1) → ② 에피소드 종료 후 `hook_pose_estimator` 서비스 1콜로
  hook pose가 행어 목표 pose 허용오차(초기값 ±3 cm / ±15°) 내인지 자동 판정 (v2, 구현항목 5).
  에피소드 시작 pose는 이미 `zivid_data/*_data.bin`으로 저장되는 인프라가 있으므로 종료 시에도
  동일 서비스 호출이면 됨.
- FT 임계/E-STOP은 Episode Manager의 기존 FAULT 기준을 그대로 사용 (별도 정의 없음).

## 4. 실험 프로토콜
1. **Sanity (로봇 5–10 ep)**: SELECT=top1(가치 균일)로 주행 — softmin 대비 거동 확인.
   지표: 성공률, FT 이벤트, jerk. 문제 시 H를 4로 줄이거나 top-2 blend.
2. **Baseline (20–30 ep)**: SELECT=softmin, 필터 뱅크. 성공률/시간 분포 확보.
3. **RL (30–50 ep)**: SELECT=thompson, 매 5 ep마다 가치 업데이트.
   중간 점검: 학습된 v의 분산 (퍼지지 않으면 = 실패 원인이 선택이 아닌 다른 곳이라는 진단).
4. **판정**: 성공률 +10%p 또는 time-to-success −20% 이상이면 Stage 2로.

## 5. 로드맵 (v1 이후 — 데이터 예산 순)
- **v1 (현재, ~30 ep)**: 가치 테이블 + Thompson — 나쁜 데모 구간 회피. 뉴럴넷 없음.
- **v1.5 (공짜, v1과 병행)**: **성공 rollout을 뱅크에 append** (non-parametric self-imitation).
  학습 없이 정책이 개선되고, 고정 뱅크 RL이 못 하는 "회복 커버리지"를 유일하게 해결.
  성공 에피소드만 넣어 convex-hull 안전성 유지.
- **v2 (~100 ep)**: 뉴럴 re-ranker `Q(e, e_i, phase_i, d_i, 최근 선택 이력)` — 미방문 엔트리
  일반화 + 이력 입력으로 aliasing 구조적 해소. top-k 중 선택이라는 이산·안전 구조는 유지,
  테이블은 prior/라벨로 재사용. (AWR식 또는 성공 BCE)
- **v3 (정밀도 병목이 입증될 때만)**: bounded residual RL (±2cm) — 데모에 없는 미세 보정.
  convex-hull 논리 일부를 포기하므로 hook 허용오차 증거 없이는 착수하지 않음.
- 인코더 RL fine-tuning은 전 단계에서 범위 밖 (retrieval 안전성 논리 보존).

## 6. 리스크와 완화
- **top-1 노이즈**: 오프라인 L1 기준 k=1은 k=32보다 나쁨(0.0770 vs 0.0633, 단발 기준).
  → temporal ensemble 유지 + H-tick 커밋 + Ruckig. Sanity 단계에서 정량 확인.
- **Aliasing (부분관측)**: prog prior + 필터 뱅크로 완화. 잔존 시 state에 FK EEF pose 추가.
- **회복 불가**: 실패 = 데모 튜브 이탈이라 뱅크에 회복 데이터가 없음. 이 플랜의 목표는
  "실패로 이어지는 분기 회피"이지 회복 학습이 아님을 명시.
- **credit 희석**: §2의 비대칭 크레딧(실패=recency 감쇠)으로 해결. 남는 한계: 이른 원인
  (나쁜 접근각)이 늦게 발현되는 실패는 감쇠 꼬리로만 잡힘 — v2 확장 후보: 실패 순간의 쿼리
  임베딩과 가까운 뱅크 엔트리에 임베딩-근접 기반 벌점 (선택 이력과 무관하게 일반화).
- **비정상 정지 라벨**: E-STOP/FAULT 에피소드는 R=0이되 마지막 구간만 강한 페널티.

## 7. 성공 시나리오의 모습
학습 후 v가 낮아진 엔트리들을 보면 "어느 데모의 어느 구간이 현장에서 재현이 안 되는가"가
그대로 나옴 — 정책 개선과 동시에 데이터셋 진단 도구가 됨. loop-break/idle-filter가 휴리스틱으로
하던 일을 보상 기반으로 대체하는 것이 이 플랜의 최종 상태.
