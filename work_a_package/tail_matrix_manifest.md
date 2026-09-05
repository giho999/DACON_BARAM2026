# TASK-G12TAIL — tail 변환 전량 매트릭스 매니페스트 (27종)

> 생성: 2026-08-09 · 작업 루트: `/home/gpu_04/DACON_baram2026/`
> 지시: `python3 work_a_package/make_tail_probe.py` 기반 λ∈{1.00,1.20,1.35} × 3그룹 = 27종 전량
> 빌더: `work_a_package/build_tail_matrix.py` (재현 스크립트, JSON 결과 `tail_matrix_results.json`)
> base: `submission_A_m1v2_coordBA.csv` (LB 0.65465, 67위)

---

## 0. 파라미터 (고정)

| 그룹 | 앵커 a | λ 사다리 |
|---|---|---|
| kpx_group_3 | **0.70** | {1.00, 1.20, 1.35} |
| kpx_group_2 | **0.80** | {1.00, 1.20, 1.35} |
| kpx_group_1 | **0.80** | {1.00, 1.20, 1.35} |

- 파일명 규칙: `submission_tail_g3l{λ}_g2l{λ}_g1l{λ}.csv` — **λ=1.00 그룹은 이름에서 생략**
  (예: g3=1.20,g2=1.20,g1=1.00 → `submission_tail_g3l120_g2l120.csv`; 전부 1.00 → `submission_tail.csv`)
- 변환: `make_tail_probe.transform` (단조·순서보존, "변경 행만 재포맷, 나머지 원문 유지")

---

## 1. 27종 md5 표

| # | 파일 | λ3/λ2/λ1 | 변경행(G3/G2/G1) | md5 |
|---|---|---|---|---|
| 1 | `submission_tail.csv` | 1.00/1.00/1.00 | 0 / 0 / 0 | `9f792b404d67e3377003e0264424991a` |
| 2 | `submission_tail_g1l120.csv` | 1.00/1.00/1.20 | 0 / 0 / 852 | `ed1e5f7fab9443cf1e87fc7008da016d` |
| 3 | `submission_tail_g1l135.csv` | 1.00/1.00/1.35 | 0 / 0 / 852 | `16d3a6c3d2b311bfe69fa90338a9b032` |
| 4 | `submission_tail_g2l120.csv` | 1.00/1.20/1.00 | 0 / 1193 / 0 | `08b7eeabe27ecfcc9811312be60a955c` |
| 5 | `submission_tail_g2l120_g1l120.csv` | 1.00/1.20/1.20 | 0 / 1193 / 852 | `b82f04a7731664c546345fb4f0390235` |
| 6 | `submission_tail_g2l120_g1l135.csv` | 1.00/1.20/1.35 | 0 / 1193 / 852 | `39f5c90dfbc8b2d21d038f4613672d0a` |
| 7 | `submission_tail_g2l135.csv` | 1.00/1.35/1.00 | 0 / 1193 / 0 | `c6ecdc84d0453c38f6588f639efdddb6` |
| 8 | `submission_tail_g2l135_g1l120.csv` | 1.00/1.35/1.20 | 0 / 1193 / 852 | `38fe67fdae547dff40ece62ee10c132d` |
| 9 | `submission_tail_g2l135_g1l135.csv` | 1.00/1.35/1.35 | 0 / 1193 / 852 | `e413d9a8d3735d90afb0005e6c604701` |
| 10 | `submission_tail_g3l120.csv` | 1.20/1.00/1.00 | 1129 / 0 / 0 | `722db0bbe71adb7637c595a65f0a74f2` |
| 11 | `submission_tail_g3l120_g1l120.csv` | 1.20/1.00/1.20 | 1129 / 0 / 852 | `3d62485ee906d08d69266d14b8e8372d` |
| 12 | `submission_tail_g3l120_g1l135.csv` | 1.20/1.00/1.35 | 1129 / 0 / 852 | `38a74d592649b14dcade5715d8443acd` |
| 13 | `submission_tail_g3l120_g2l120.csv` | 1.20/1.20/1.00 | 1129 / 1193 / 0 | `360e8788327a921f18a93285393c4367` |
| 14 | `submission_tail_g3l120_g2l120_g1l120.csv` | 1.20/1.20/1.20 | 1129 / 1193 / 852 | `9b214aa5e2c0d214730a3022b22f0138` |
| 15 | `submission_tail_g3l120_g2l120_g1l135.csv` | 1.20/1.20/1.35 | 1129 / 1193 / 852 | `32156ad0964f4f15674ecd2d1add58a8` |
| 16 | `submission_tail_g3l120_g2l135.csv` | 1.20/1.35/1.00 | 1129 / 1193 / 0 | `e9b6c5b4ed0cd5326a6d4ecd6f188a93` |
| 17 | `submission_tail_g3l120_g2l135_g1l120.csv` | 1.20/1.35/1.20 | 1129 / 1193 / 852 | `7148bb40f15f21b23740ea1aa1637fdc` |
| 18 | `submission_tail_g3l120_g2l135_g1l135.csv` | 1.20/1.35/1.35 | 1129 / 1193 / 852 | `54f598587c8a1cb35c13c2424eb1a9ed` |
| 19 | `submission_tail_g3l135.csv` | 1.35/1.00/1.00 | 1129 / 0 / 0 | `e6f75fbc33f49666f42fa7852abcbf84` |
| 20 | `submission_tail_g3l135_g1l120.csv` | 1.35/1.00/1.20 | 1129 / 0 / 852 | `b02cfa85fcaada3d49579111d868c96b` |
| 21 | `submission_tail_g3l135_g1l135.csv` | 1.35/1.00/1.35 | 1129 / 0 / 852 | `a98d2f942a8ce1b689c5b43baeed14b3` |
| 22 | `submission_tail_g3l135_g2l120.csv` | 1.35/1.20/1.00 | 1129 / 1193 / 0 | `2a7821a794ba15805e423baeebd31613` |
| 23 | `submission_tail_g3l135_g2l120_g1l120.csv` | 1.35/1.20/1.20 | 1129 / 1193 / 852 | `f57e5923015bafd978343d15a802d3ce` |
| 24 | `submission_tail_g3l135_g2l120_g1l135.csv` | 1.35/1.20/1.35 | 1129 / 1193 / 852 | `9a101c298ce2c56090872b688a01dfcf` |
| 25 | `submission_tail_g3l135_g2l135.csv` | 1.35/1.35/1.00 | 1129 / 1193 / 0 | `ffb2cfb1725d43d106661624298c044b` |
| 26 | `submission_tail_g3l135_g2l135_g1l120.csv` | 1.35/1.35/1.20 | 1129 / 1193 / 852 | `b6d0f07967fcf91ddb180ea0433f0d01` |
| 27 | `submission_tail_g3l135_g2l135_g1l135.csv` | 1.35/1.35/1.35 | 1129 / 1193 / 852 | `1408df17ccd518141402d0b9e755fb89` |

**변경행 수 (그룹별, λ에만 의존)**: G3 λ≥1.20 → 1129 · G2 λ≥1.20 → 1193 · G1 λ≥1.20 → 852
(λ=1.00 → 0). G1 852 < 859(cf>0.80)는 cap(1.0000) 도달 7행의 clip 불변 때문.

---

## 2. 가드 검증 결과 (27종 전수)

| 가드 | 항목 | 결과 |
|---|---|---|
| **NULL** | `submission_tail.csv`(전 λ=1.00) md5 == base md5 | ✅ PASS (`9f792b404d67…` 일치) |
| **L1** | λ=1.00 그룹 원문 문자열 100% 미접촉 · λ>1.00 그룹 접촉 정상 | ✅ PASS (27종 전부) |
| L1 | forecast_id / kst_dtm 100% · 8760행 · 헤더 정합 · sample_submission 대응 | ✅ PASS |
| **L2** | 변경행 수 (0/852/1129/1193) · 앵커 이하 변경 == 0 | ✅ PASS (전 파일) |
| **L3** | NaN/음수/cap초과 0건 · 그룹별 Spearman ρ == 1.0 | ✅ PASS (3그룹×27종 ρ=1.0) |
| **결정성** | 동일 커맨드 재실행 시 md5 일치 (기존 G12TAIL 파일과 대조) | ✅ PASS (e.g. #13/14/22 = 기존 `g3l120_g2l120`·`g3l120_g2l120_g1l120`·`g3l135_g2l120` md5 동일) |

---

## 3. 제출 시 유의 (G12TAIL 선등록과 연동)

- **P2 후보** (8/10 G3 판독 후): #13 (`g3l120_g2l120`) / #4 (`g2l120`, G3 미채택 시) / #22 (`g3l135_g2l120`, G3 승급 시)
- **P3 후보**: #14 (`g3l120_g2l120_g1l120`) / #5 / #23
- **P4 예비** (λ1.35 승급): #15/17/18/24/26/27 등
- 전부 결정적 산출물 — LB 판정은 선등록 `PREREG_g12tail_20260810.md` 기준(ΔLB ≥ +0.0010 채택 등).
- **제출은 사용자 수행.** 자동 제출 금지.

---

## 4. 재현 커맨드

```bash
cd /home/gpu_04/DACON_baram2026 && python3 work_a_package/build_tail_matrix.py
```

- 27종 전량 재생성 + 가드 전수 재검증 + `tail_matrix_results.json` 갱신. 결정적 (무작위성 0).
- 단건 생성: `python3 work_a_package/make_tail_probe.py --src work_a_package/submission_A_m1v2_coordBA.csv --dst <out> --spec kpx_group_3:0.70:1.20 --spec kpx_group_2:0.80:1.20 ...`

---

## 5. λ<1 쇼링크 방향 지원 (2026-08-09 추가) — shrink085/090/080

- `make_tail_probe.py` 파라미터 가드를 `λ<1.0 → 차단`에서 **`λ≤0 → 차단`**으로 완화
  (λ>0이면 단조성 보존 — 실측 ρ=1.0 확인). 쇼링크(λ<1)는 상단 tail을 앵커 쪽으로 끌어내림.
- **변경행 수**: G3 1129 / G2 1193 / **G1 859** (λ>1 때 852와 다름 — λ<1에선 cap 도달 7행도
  값이 감소해 변경되므로 cf>a 전부 859행 변경. 정상).

| 파일 | λ(3/2/1) | 변경행(G3/G2/G1) | md5 |
|---|---|---|---|
| `submission_tail_shrink085.csv` | 0.85/0.85/0.85 | 1129 / 1193 / 859 | `ea96a9e6d12848a589255708b338ceb1` |
| `submission_tail_shrink090.csv` | 0.90/0.90/0.90 | 1129 / 1193 / 859 | `110d1d7af5914ec52288839d4d409cc9` |
| `submission_tail_shrink080.csv` | 0.80/0.80/0.80 | 1129 / 1193 / 859 | `77fc754fbf516a3e38b5779a526e4ae2` |

- 가드: **NULL**(전 λ=1.00 = `submission_tail.csv` == base md5 `9f792b40…`) ✅ · **L1** ✅ (id/dtm/헤더/행수/샘플 정합) ·
  **L2** ✅ (변경행=cf>a 전수, 앵커 이하 0) · **L3** ✅ (NaN/음수/cap초과 0, **3그룹 ρ=1.0** — λ>0 단조성 보존 실측)
- SHA-256: shrink085 `9eec879a…` · shrink090 `d5e0c4db…` · shrink080 `d1429cb6…`
- 재현: `python3 work_a_package/make_tail_probe.py --src work_a_package/submission_A_m1v2_coordBA.csv --dst work_a_package/submission_tail_shrink085.csv --spec kpx_group_3:0.70:0.85 --spec kpx_group_2:0.80:0.85 --spec kpx_group_1:0.80:0.85` (λ만 0.90/0.80으로 바꿔 090/080)

---

*base md5 `9f792b404d67e3377003e0264424991a` · SHA-256 `885666e2a8aa6070eb52d3fb20ef147a8f2d488c9924969af44984a62c86d429`*
