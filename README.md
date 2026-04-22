# HighlightApp

저조도 이미지를 AI로 밝게 보정하는 Android 앱. Illumination-Adaptive-Transformer(IAT) 모델을
온디바이스 TFLite로 실행하고, full-resolution 원본의 선명도를 유지하면서 밝기만 끌어올립니다.

- **모델**: IAT (LOL-v1 가중치, 91 K parameters, ≈ 420 KB TFLite)
- **런타임**: TFLite GPU delegate (FP16 자동 승격) → CPU XNNPACK fallback
- **UI**: Jetpack Compose + Material 3, min SDK 28 / target SDK 35
- **플로우**: 사진 선택 → 자동 enhancement → 강도 슬라이더 → 갤러리 저장

> 핵심 개념 / 파이프라인 / 버그 히스토리는 [`docs/OVERVIEW.md`](docs/OVERVIEW.md) 참고.

---

## 주요 특징

- **Gain-map full-res 복원**: 512×512 모델 출력을 직접 업스케일하지 않고, `(enhanced / original)`
  저주파 비율만 bilinear로 업스케일해서 원본 해상도에 곱함. → 디테일 그대로, 밝기만 적용.
- **Gray-world White Balance**: IAT가 일부 입력에서 만드는 분홍/마젠타 cast를 채널 평균 기반으로 중화.
- **Intensity slider**: 원본↔enhanced를 알파 블렌드. 0 ~ 100 %, 기본 75 %.
- **No backend**: 모든 처리는 기기 내. 인터넷/권한 불필요 (Photo Picker + MediaStore).

## 스크린샷 / 비교

| 원본 | 직접 업스케일 + WB (soft) | Gain-map + WB (sharp) |
|---|---|---|
| 원본 1024×682 저조도 이미지 | 밝기는 올라가지만 blur | 선명도 유지 + 밝기 |

전체 샘플 비교는 `python convert/compare_gainmap.py` 실행 시
`convert/artifacts/compare_gainmap/` 에 저장됩니다.

## 빌드

### 요구사항

- JDK 17
- Android SDK (platform-tools + cmdline-tools)
- Kotlin 2.0.21 / AGP 8.7.3 (Gradle wrapper 포함)

### APK 빌드

```bash
# JDK 17 경로 지정
export JAVA_HOME=/path/to/jdk-17
./gradlew assembleDebug
```

결과물: `app/build/outputs/apk/debug/app-debug.apk` (≈ 46 MB, dex + TFLite GPU delegate 포함)

### 설치

```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

## 모델 변환 (선택, 재현용)

TFLite 모델 `app/src/main/assets/iat_enhance.tflite` 는 레포에 포함되어 있어 별도 변환 없이 빌드됩니다.
재생성하려면:

```bash
# 1. 상위 레포 clone + LOL-v1 가중치 준비
git clone https://github.com/cuiziteng/Illumination-Adaptive-Transformer convert/iat_source
# convert/weights/best_Epoch_lol_v1.pth 배치 (원 저자 릴리스)

# 2. 변환
pip install -r convert/requirements.txt
python convert/export_onnx.py        # IAT -> ONNX (per-channel color matrix unroll)
python convert/onnx_to_tflite.py     # ONNX -> TFLite (-tb tf_converter)
python convert/verify_parity.py      # PyTorch ↔ TFLite parity (PSNR)

# 3. Android 앱에 배포
cp convert/artifacts/iat_enhance.tflite app/src/main/assets/
```

## ONNX → TFLite 변환 시 반드시 알아야 할 함정

두 개의 `onnx2tf` 버그가 IAT를 조용히 망가뜨립니다:

1. **`flatbuffer_direct` 백엔드는 global branch transformer를 잘못 변환**
   일부 입력에서 완전 검정 출력.
   → `-tb tf_converter` 로 해결.

2. **`image @ color.T` matmul 축을 잘못 잡아 G 채널을 ~50% 억제**
   → matmul을 per-channel 스칼라 곱셈으로 풀어서 export (`convert/export_onnx.py::ExportWrapper`).

PyTorch ↔ ONNX 는 항상 정확히 일치. 버그는 둘 다 ONNX → TFLite 단계에서 발생.
자세한 내용은 [`docs/OVERVIEW.md`](docs/OVERVIEW.md) §4.

## 디렉토리 구조

```
app/                      Android 앱 (Kotlin + Compose)
  └ src/main/
     ├ assets/iat_enhance.tflite       최종 모델
     ├ kotlin/.../inference/           IATInterpreter, ImagePipeline
     └ kotlin/.../ui/                  HomeScreen, ResultScreen

convert/                  PyTorch → TFLite 변환 파이프라인 (Python)
  ├ export_onnx.py, onnx_to_tflite.py, verify_parity.py
  └ test_lowlight.py, compare_gainmap.py, compare_strategies.py

docs/OVERVIEW.md          핵심 개념, 파이프라인, 함정 히스토리
```

## Acknowledgments

- [Illumination-Adaptive-Transformer](https://github.com/cuiziteng/Illumination-Adaptive-Transformer) —
  Cui 등, ECCV 2022. 원 모델 및 LOL-v1 pretrained weights.
- [`onnx2tf`](https://github.com/PINTO0309/onnx2tf) — ONNX → TFLite 컨버터.

## License

- 앱 소스: MIT
- IAT 모델 weights: 상위 레포의 라이선스를 따름 (별도 확인 필요)
