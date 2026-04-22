# HighlightApp — 핵심 키워드 정리

> 저조도 사진을 한 번 탭으로 밝게 바꿔주는 안드로이드 AI 필터 앱.

---

## 1. 목적

- **저조도 이미지 개선**을 모바일에서 실시간에 가깝게 제공
- 백엔드 의존 없이 **온디바이스 추론**으로 완결
- "사진 선택 → 처리 → 강도 조절 → 저장" 네 단계의 **심플한 UX**

## 2. 핵심 기술 스택

| 레이어 | 기술 |
|---|---|
| Model | **IAT** (Illumination-Adaptive-Transformer, LOL-v1 weights, 91 K params) |
| 변환 | PyTorch → ONNX (opset 17) → TFLite (`onnx2tf` `tf_converter` 백엔드) |
| 런타임 | TFLite GPU delegate (FP16 자동 승격) → XNNPACK CPU fallback |
| UI | Kotlin + Jetpack Compose + Material 3 |
| 입출력 | Photo Picker API, MediaStore JPEG save (런타임 권한 불필요) |

## 3. 파이프라인 (Android, `ImagePipeline.kt`)

```
원본 Bitmap ───────────────────────────────────────┐
    │                                              │
    ▼                                              │
 letterbox 512×512 ─► IAT (TFLite, FP32 I/O) ──► content crop (contentW×contentH)
                                                   │
                                                   ▼
                                         gray-world WB (low-res)
                                                   │
                                                   ▼
                                ┌────── gainMapEnhance ──────┐
                                │                            │
                                │ • 원본 → 동일 저해상도 다운샘플  │
                                │ • 둘 다 full-res bilinear 업  │
                                │ • gain = enh / origBlur        │
                                │   (채널별, [0, 20] 클램프)   │
                                │ • 결과 = 원본 × gain         │
                                └────────────────────────────┘
                                                   │
                                                   ▼
                                     UI 강도 슬라이더로 원본 alpha-blend
                                                   │
                                                   ▼
                                           MediaStore 저장 (JPEG)
```

핵심 포인트:
- **Gain-map 복원**: IAT 출력을 바로 업스케일하면 고주파 디테일이 날아감. 대신 (enhanced / original) 저주파 비율만 업스케일해서 full-res 원본에 곱함 → **원본 선명도 유지 + 밝기만 적용**
- **Gray-world WB**: IAT가 가끔 분홍 cast를 띄는 것을 채널 평균으로 중화
- **Intensity slider**: 0 ~ 100 %, 기본값 75 %. 저장 시 blended 결과를 JPEG로 인코딩

## 4. 모델 변환 — 숨은 함정 2개

### Bug 1 — `flatbuffer_direct` 백엔드의 global branch 손상
IAT의 `global_net`(transformer 기반 3×3 color matrix + gamma 예측기)이 TFLite에서 완전히 다른 값으로 컨버전됨. 결과: 일부 입력에서 **검정 출력** (예: `kr_seoul_river.jpg`).
**Fix**: `python -m onnx2tf -tb tf_converter` (classic 백엔드 사용)

### Bug 2 — 3×3 color matrix matmul 축 오류
`tf_converter`로 바꿔도 `image @ color.T` matmul 축을 잘못 잡아 **G 채널이 ~50% 억제**됨 → 노란-초록 색왜곡.
**Fix**: matmul을 per-channel 스칼라 곱셈으로 풀어서 export. `convert/export_onnx.py`의 `ExportWrapper` 참고.

### 검증
```
|torch - tflite| < 0.000004   (사실상 bit-exact)
```

## 5. 파일 지도

```
app/                                      # Android 앱 소스
├── src/main/assets/iat_enhance.tflite   # 최종 모델 (≈ 420 KB, FP32 weights)
├── src/main/kotlin/.../inference/
│   ├── IATInterpreter.kt                # GPU delegate + XNNPACK fallback
│   └── ImagePipeline.kt                 # letterbox / gain-map / WB / blend
└── src/main/kotlin/.../ui/
    ├── HomeScreen.kt                    # 사진 선택
    └── ResultScreen.kt                  # 강도 슬라이더 + 저장

convert/                                  # 학습된 PyTorch 가중치를 TFLite로 변환
├── export_onnx.py                       # IAT → ONNX (per-channel color unroll)
├── onnx_to_tflite.py                    # ONNX → TFLite (-tb tf_converter)
├── verify_parity.py                     # PyTorch ↔ TFLite PSNR 체크
└── test_lowlight.py, compare_gainmap.py # 시각 비교 스크립트

docs/OVERVIEW.md                          # 이 문서
```

## 6. 핵심 키워드

- `Illumination-Adaptive-Transformer` · `onnx2tf` · `tf_converter backend`
- `per-channel color matrix unroll` · `gain-map upscale` · `gray-world WB`
- `TFLite GPU delegate` · `FP32 I/O + runtime FP16 promotion`
- `Jetpack Compose` · `Photo Picker` · `MediaStore`
- `letterbox 512×512` · `intensity alpha-blend`
