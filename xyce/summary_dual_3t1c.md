# Dual 3T1C IGZO Circuit Simulation Summary

## 1. IGZO TFT Model

### Parameters (scipy refit, K=5.0889)
| Symbol | Value |
|--------|-------|
| L | 6.9526 |
| K | 5.0889 |
| V0 (Vth) | 0.1102 V |
| B | −13.1232 |
| VSAT | 1.287 V |
| SLOPE | 2.307×10⁻⁷ A/V |

### I–V Formula (bidirectional)
```
I_single(Vgs) = [10^(B + L / (1 + exp(-K*(min(Vgs, VSAT) - V0)))) + max(0, Vgs-VSAT)*SLOPE] × 1e9  [nA]

I_net(drain→source) = I_single(Vgs) - I_single(Vgd)
```
- Subcircuit: `igzo_smooth_k5.sub`  
- Port order: `XDEV drain gate source IGZO_SMOOTH_K5`
- Vth override: `XDEV drain gate source IGZO_SMOOTH_K5 PARAMS: V0=<new_vth>`

---

## 2. Single 3T1C Circuit (M0)

### Circuit Structure
```
V1(0V) ─── M2(gate=V2=3V) ─── mid ─── M0(gate=SN) ─── V̄5
                                  └─── M1(gate=V3=−3V) ─── SN
                                         │
                                    capacitor
                                         │
                                        V4
```

- **M0**: gate=SN, one terminal=mid, other terminal=V̄5
- **M1**: always OFF (gate=−3V), isolates SN from mid in Phase 2
- **M2**: always ON (gate=3V), clamps mid ≈ 0V (V1=0V side)
- **Capacitor**: between SN and V4

### Phase 1 — Vth Storage

V̄5 = −V5 applied. Circuit charges SN until M0 cuts off:

```
Vgs_M0 = V(SN) − V(V̄5) = V(SN) + V5 = Vth
→  VSN_stored = Vth − V5
```

**Example** (V5=1V, Vth=0.1102V):
```
VSN_stored = 0.1102 − 1.0 = −0.8898 V
```

### Phase 2 — Current Drive

V4 bootstraps onto SN via capacitor. V̄5 switches to VDD:

```
VSN_final = VSN_stored + V4 = (V4 − V5) + Vth
```

Terminal conditions: V̄5 = VDD, V1 = 0V

```
Vgd_eff = VSN_final − 0V = V4 − V5 + Vth   ← effective gate voltage (wrt 0V)
Vgs_eff = VSN_final − VDD = V4 − V5 + Vth − VDD

→  Vth cancelled: effective drive = (V4 − V5)
```

**Linear regime formula** (K absorbs 1/2):
```
I_M0 = K · VDD · (2(V4−V5) − VDD)
```

**Turn-on threshold**: V4 = V5 (Vth-independent) ✓

**Xyce BSN setting**:
```spice
BSN  sn  0  V={V(v4n) + (Vth - V5)}   ; = V4 - 0.8898 for V5=1V
V5B  v5bn 0  DC 1.0                    ; V̄5 = VDD in Phase 2
XM0  mid  sn  v5bn  IGZO_SMOOTH_K5
```

---

## 3. Single 3T1C Circuit (M3)

### Symmetric counterpart of M0

```
V1(0V) ─── M5(gate=V2=3V) ─── mid ─── M3(gate=SN) ─── V̄4
                                  └─── M4(gate=V3=−3V) ─── SN
                                         │
                                    capacitor
                                         │
                                        V5
```

### Phase 1 — Vth Storage (M3)

V̄4 = −V4 applied:
```
Vgs_M3 = V(SN) + V4 = Vth
→  VSN_stored = Vth − V4
```

### Phase 2 — Current Drive (M3)

V5 bootstraps:
```
VSN_final = (V5 − V4) + Vth

→  effective drive = (V5 − V4)
I_M3 = K · VDD · (2(V5−V4) − VDD)
```

**Turn-on threshold**: V5 = V4 (same crossing point as M0) ✓

**Xyce BSN setting** (V5=1V fixed, V4 swept):
```spice
BSN  sn  0  V={(V5 + Vth) - V(v4n)}   ; = 1.1102 - V4
V4B  v4bn 0  DC 1.0                    ; V̄4 = VDD in Phase 2
XM3  mid  sn  v4bn  IGZO_SMOOTH_K5
```

---

## 4. Dual 3T1C — Combined Output

### Circuit

Two circuits share V1 output rail:

```
         V1 ────────────────────────────
          |                            |
         M2                           M5
          |                            |
         mid_M0   mid_M3
          |                            |
         M0 (gate=SN_M0)             M3 (gate=SN_M3)
          |                            |
     V̄5=VDD (Phase2)            V̄4=VDD (Phase2)
```

| | M0 | M3 |
|--|----|----|
| Cap reference | V4 | V5 |
| Source (Phase 2) | V̄5 = VDD | V̄4 = VDD |
| Active region | V4 > V5 | V5 > V4 |
| Formula | I ∝ (V4−V5)·VDD | I ∝ (V5−V4)·VDD |

### Key Result

- **Turn-on**: both circuits cross at V4 = V5 (threshold-independent)
- **Total output**: I_total = I_M0 + I_M3 — always positive for all V4, V5 ✓
- **V-shape**: I_total minimum at V4 = V5, rises symmetrically on both sides

---

## 5. Vth Shift Analysis (+0.5V)

| Parameter | Original | Shifted |
|-----------|----------|---------|
| Vth (V0) | 0.1102 V | 0.6102 V |
| BSN_M0 | V4 − 0.8898 | V4 − 0.3898 |
| BSN_M3 | 1.1102 − V4 | 1.6102 − V4 |
| Turn-on (V4=V5) | **unchanged** ✓ | **unchanged** ✓ |
| M0 kink (Vgs=VSAT) | V4 = 2.177V (out of range) | V4 = **1.677V** |
| M3 kink (Vgs=VSAT) | V4 = −0.177V | V4 = **0.323V** |

### Conclusion

- **Subthreshold / saturation 구간**: Vth shift 완전 상쇄 — 두 곡선 완전히 겹침
- **Turn-on 기준점**: V4=V5에서 동일 — 3T1C가 Vth를 정확히 보상
- **차이**: kink(Vgs=VSAT) 위치만 이동 → linear regime 진입 타이밍 변화
- **Total (I_M0+I_M3)**: Vth shift에도 거의 변화 없음 — 회로 레벨 보상 효과

---

## 6. Simulation Files

| File | Description |
|------|-------------|
| `igzo_smooth_k5.sub` | IGZO TFT Xyce subcircuit |
| `run_phase2_m0.py` | M0 Phase 2 Xyce simulation (V5=1V, V4 swept) |
| `run_phase2_m3.py` | M3 Phase 2 Xyce simulation (V5=1V, V4 swept) |
| `run_phase2_m0_vthshift.py` | M0 Phase 2 with Vth+0.5V shift |
| `run_dual_phase2_vthshift.py` | Dual M0+M3 Vth comparison (Xyce) |
| `plot_dual_phase2.py` | Dual M0+M3 combined plot (Xyce) |
| `plot_dual_phase2_python.py` | Dual M0+M3 combined plot (Python model) |

| Output PNG | Description |
|-----------|-------------|
| `phase2_m0_log.png` | M0 single: Xyce + Python, V4 vs log\|I_M0\| |
| `phase2_m3_log.png` | M3 single: Xyce + Python, V4 vs log\|I_M3\| |
| `phase2_m0_vthshift_log.png` | M0: original vs Vth+0.5V shift |
| `dual_phase2_log.png` | M0+M3+total (Xyce solid, Python dashed) |
| `dual_phase2_vthshift_log.png` | Dual: original vs Vth+0.5V (Xyce) |
| `dual_phase2_log_pyt.png` | M0+M3+total (Python model only) |
| `dual_phase2_vthshift_log_pyt.png` | Dual: original vs Vth+0.5V (Python) |

---

## 7. 핵심 물리 요약

```
Phase 1:  VSN = Vth − Vref       (Vref = V5 for M0, V4 for M3)
Phase 2:  VSN_final = Vcontrol − Vref + Vth
          Vgd_eff   = Vcontrol − Vref + Vth  (wrt 0V terminal)
                    = (Vcontrol − Vref) + Vth

          → Vth cancelled by 3T1C storage mechanism
          → I ∝ f(Vcontrol − Vref) × VDD
```

| Circuit | Vcontrol | Vref | Active when |
|---------|----------|------|-------------|
| M0 | V4 | V5 | V4 > V5 |
| M3 | V5 | V4 | V5 > V4 |
| Dual | — | — | always (I_M0 + I_M3 > 0) |
