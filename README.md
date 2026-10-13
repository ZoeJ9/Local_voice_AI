# 회의록 자동화 시스템

한국어 + 영어 혼용 회의 녹음을 완전 로컬 환경에서 자동으로 회의록으로 변환합니다.

## 처리 파이프라인

```
🎤 녹음 파일 (mp3 / wav / m4a / mp4) 또는 마이크 실시간 녹음
        ↓
[ffmpeg]   mp3/m4a/mp4 → wav 변환 (모노 16kHz)
        ↓
[whisperx - large-v3-turbo]   STT + 단어 단위 타임스탬프 정렬 (한/영 자동 감지)
        ↓
[whisperx - DiarizationPipeline]   화자 분리 + 자동 매핑 (Speaker 1, 2, ...)
        ↓
[Ollama qwen2.5:14b]   회의록 정리 & 요약
        ↓
📄 Obsidian vault → YYYY-MM-DD_회의록.md
```

### whisperx 내부 처리 흐름

```
whisperx.load_audio()          오디오 로드
        ↓
model.transcribe()             STT (언어 자동 감지)
        ↓
whisperx.align()               단어 단위 정밀 타임스탬프 정렬
        ↓
DiarizationPipeline()          화자 분리 (내부적으로 pyannote 사용)
        ↓
assign_word_speakers()         STT 결과 + 화자 정보 자동 매핑
```

## 환경 요구사항

| 항목 | 사양 |
|------|------|
| OS | Windows |
| RAM | 32GB 이상 권장 |
| Python | 3.11 (conda voice_ai 환경) |
| Ollama | 백그라운드 실행 중 |
| 모델 | qwen2.5:14b (Ollama), large-v3-turbo (최초 실행 시 자동 다운로드) |
| ffmpeg | 별도 설치 필요 (아래 참고) |

## 설치

### 1. conda 환경 활성화
```bash
conda activate voice_ai
```

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

### 3. ffmpeg 설치 (mp3/m4a/mp4 변환 + whisperx 오디오 처리)

conda로 설치가 안 될 경우 공식 빌드를 직접 다운로드:
- https://github.com/BtbN/ffmpeg-builds/releases 에서 `ffmpeg-master-latest-win64-gpl-shared.zip` 다운로드
- 압축 해제 후 `bin` 폴더 경로를 [meeting_transcriber.py](meeting_transcriber.py)의 `_get_ffmpeg_path()` 함수에 지정

### 4. Ollama 모델 준비
```bash
ollama pull qwen2.5:14b
```

### 5. HuggingFace 토큰 발급 (화자 분리용 - 최초 1회 인터넷 필요)
1. https://huggingface.co/settings/tokens 접속
2. New token 생성 → **"Access public gated repositories"** 권한 활성화
3. 아래 모델 페이지에서 이용 약관 동의:
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-community-1

### 6. config.yaml 설정
```bash
cp config.yaml.example config.yaml
# config.yaml 열어서 경로와 토큰 입력
```

## config.yaml 설정

```yaml
# Obsidian vault 경로 (슬래시 / 사용, 백슬래시 \ 사용 금지)
obsidian_vault_path: "C:/Users/YourName/ObsidianVault/회의록"

# HuggingFace 토큰 (화자 분리용)
huggingface_token: "hf_xxxxxxxxxxxxxxxxxxxx"

# 화자 이름 매핑 (회의 후 수정 가능, 비워두면 Speaker 1/2/3 자동 부여)
speaker_names:
  SPEAKER_00: "홍길동"
  SPEAKER_01: "김철수"
  SPEAKER_02: "이영희"

ollama:
  host: "http://localhost:11434"
  model: "qwen2.5:14b"
  timeout: 300
```

> **주의**: Obsidian 경로에 백슬래시(`\`) 사용 시 YAML 파싱 오류 발생. 반드시 슬래시(`/`) 사용

## 사용법

### 녹음 파일 변환
```bash
python meeting_transcriber.py --file "회의녹음.mp3"
```

### 실시간 마이크 녹음
```bash
python meeting_transcriber.py --mic
# Enter 키로 녹음 종료
```

### 화자 이름 변경
`config.yaml`의 `speaker_names` 수정 후 재실행

## 출력 파일 형식

```markdown
# 회의록 - 2025-01-15

## 📋 중요 내용 요약
...

## 🔧 Technical Detail 요약
...

## ✅ 액션 아이템
### 홍길동
- [ ] ...

### 김철수
- [ ] ...

## 📝 전체 스크립트
**홍길동**: ...
**김철수**: ...
```

파일 저장 위치: `{obsidian_vault_path}/2025-01-15_회의록.md`
같은 날짜 파일 존재 시: `2025-01-15_회의록_2.md` 자동 넘버링

## 파일 구조

```
Local Project/
├── meeting_transcriber.py        # 메인 실행 파일
├── config.yaml                   # 설정 파일 (git에 올리지 말 것)
├── config.yaml.example           # 설정 파일 예시 (토큰 없음)
├── requirements.txt
├── README.md
└── .vscode/
    └── settings.json             # voice_ai 환경 지정
```

## 보안 주의사항

- `config.yaml`은 HuggingFace 토큰 + Obsidian 경로 포함 → **절대 git에 커밋하지 말 것** (`.gitignore`에 포함됨)
- 회의 내용은 터미널 로그에 출력되지 않음 (진행 상태만 표시)
- 녹음 임시 파일(`temp.wav`)은 처리 완료 후 자동 삭제 (오류 발생 시에도)

## 주요 기능

| 기능 | 설명 |
|------|------|
| STT + 화자 분리 통합 | whisperx 단일 파이프라인으로 처리 |
| 한/영 혼용 | 언어 자동 감지, 혼용 지원 |
| 화자 자동 매핑 | 타임스탬프 기반 정확한 화자 매핑 |
| 중복 방지 | 같은 날짜 파일 자동 넘버링 |
| Obsidian 연동 | vault에 바로 저장, 마크다운 호환 형식 |
| 보안 | 회의 내용 로그 미출력, 토큰 분리 저장 |

## v2 - SenseVoice + qwen3 (실험적)

`meeting_transcriber_v2.py`는 기존 v1과 별개로 동작하는 실험적 버전입니다. v1(`meeting_transcriber.py`)은 수정되지 않았으며 그대로 사용 가능합니다.

```
🎤 녹음 파일 또는 마이크 실시간 녹음
        ↓
[ffmpeg]   wav 변환 (모노 16kHz)
        ↓
[pyannote.audio Pipeline]   화자 분리 (먼저 실행)
        ↓
[SenseVoiceSmall (funasr)]   화자 구간별 STT + 감정/이벤트 태그 추출
        ↓
[Ollama qwen3:14b]   회의록 정리 & 요약
        ↓
📄 Obsidian vault → YYYY-MM-DD_회의록.md
```

### v1과의 차이

| 항목 | v1 | v2 |
|------|----|----|
| STT 엔진 | whisperx (large-v3-turbo) | SenseVoiceSmall (funasr) |
| 처리 순서 | STT+화자분리 통합 후 매핑 | 화자분리 먼저 → 구간별 STT |
| 감정/이벤트 태그 | 없음 | 웃음, 한숨 등을 스크립트 줄 끝에 표시 |
| 요약 모델 | qwen2.5:14b | qwen3:14b |

### 추가 설치

```bash
pip install funasr modelscope
pip install torch --index-url https://download.pytorch.org/whl/cpu
ollama pull qwen3:14b
```

`SenseVoiceSmall`(`iic/SenseVoiceSmall`)은 HuggingFace가 아닌 **ModelScope**에서 최초 실행 시 자동 다운로드됩니다 (별도 토큰 불필요).

### config.yaml에 추가 필요

`config.yaml`(실제 설정 파일, git에 커밋되지 않음)에 아래 섹션을 직접 추가하세요. v1의 `ollama:` 섹션은 그대로 두고 별도로 추가합니다.

```yaml
ollama_v2:
  host: "http://localhost:11434"
  model: "qwen3:14b"
  timeout: 300
```

### 실행

```bash
python meeting_transcriber_v2.py --file "회의녹음.mp3"
python meeting_transcriber_v2.py --mic
```

### 문제 해결 (v2 전용)

**`<think>` 태그가 회의록에 섞여 나옴**
- qwen3는 기본적으로 추론 과정을 `<think>...</think>`로 감싸 출력할 수 있습니다. 코드에서 `think: false`를 요청하지만 구버전 Ollama는 이 옵션을 무시할 수 있습니다. Ollama를 최신 버전으로 업데이트하세요. (코드에도 방어적으로 `<think>` 블록을 제거하는 처리가 있습니다.)

**SenseVoice 모델 다운로드 실패**
- ModelScope 접속이 필요합니다 (최초 1회). 방화벽/프록시 환경이면 접속 가능 여부를 확인하세요.

## 문제 해결

**Ollama 연결 실패**
```bash
ollama list  # Ollama 실행 중인지 확인
```

**ffmpeg 오류 (mp3/m4a 변환 실패 또는 whisperx 오디오 로드 실패)**
- `_get_ffmpeg_path()` 함수에 ffmpeg.exe 경로가 올바르게 지정되어 있는지 확인

**HuggingFace 403 오류**
- 토큰의 "Access public gated repositories" 권한 확인
- 위 3개 모델 페이지에서 이용 약관 동의 여부 확인
- 최초 실행 시에만 인터넷 필요, 이후 완전 로컬 동작

**lzma DLL 오류 (Windows)**
```bash
# miniconda base 환경에서 voice_ai로 DLL 복사
cp C:\Users\{username}\miniconda3\Library\bin\liblzma.dll \
   C:\Users\{username}\miniconda3\envs\voice_ai\Library\bin\
```

**YAML 파싱 오류 (config.yaml)**
- Obsidian 경로에 백슬래시(`\`) 대신 슬래시(`/`) 사용