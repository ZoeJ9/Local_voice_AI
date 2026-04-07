# ============================================================
# 회의록 자동화 시스템
# whisperx (STT + 화자분리 통합) + Ollama (qwen2.5:14b)
# 완전 로컬 실행 - 인터넷 불필요 (모델 최초 다운로드 제외)
# ============================================================

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import yaml
from scipy.io import wavfile

# ============================================================
# 설정 파일 로드
# ============================================================

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def load_config() -> dict:
    """config.yaml 파일을 읽어 설정값 반환"""
    if not CONFIG_PATH.exists():
        print("[오류] config.yaml 파일이 없습니다. config.yaml.example을 복사해서 설정하세요.")
        sys.exit(1)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 필수 항목 검증
    if not config.get("obsidian_vault_path"):
        print("[오류] config.yaml에 obsidian_vault_path가 설정되지 않았습니다.")
        sys.exit(1)
    if not config.get("huggingface_token"):
        print("[오류] config.yaml에 huggingface_token이 설정되지 않았습니다.")
        sys.exit(1)

    return config


# ============================================================
# 오디오 입력 처리
# ============================================================

SUPPORTED_FORMATS = [".mp3", ".wav", ".m4a", ".mp4"]
TEMP_WAV_PATH = Path(__file__).parent / "temp.wav"
CHUNK_DURATION_SEC = 600  # 청크 분할 기준: 10분


def _get_ffmpeg_path() -> str:
    """conda 환경의 ffmpeg 실행 파일 경로 반환"""
    # conda 환경 기준 경로 (Windows)
    ffmpeg_path = Path(r"C:\Users\mint9\Documents\Local Project\ffmpeg-master-latest-win64-gpl-shared\bin\ffmpeg.exe")
    if ffmpeg_path.exists():
        return str(ffmpeg_path)
    # PATH에서도 탐색
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    print("[오류] ffmpeg를 찾을 수 없습니다. 다음 명령어로 설치하세요:")
    print("  conda install -c conda-forge ffmpeg -n voice_ai")
    sys.exit(1)


def convert_to_wav(input_path: Path) -> Path:
    """mp3/m4a/mp4 파일을 ffmpeg로 직접 wav 변환 (모노 16kHz)"""
    import subprocess
    print(f"[변환 중] {input_path.name} → wav 형식으로 변환 중...")

    ffmpeg = _get_ffmpeg_path()

    # ffmpeg로 직접 변환: 모노, 16kHz, 기존 temp.wav 덮어쓰기
    cmd = [
        ffmpeg,
        "-y",                      # 덮어쓰기 허용
        "-i", str(input_path),
        "-ac", "1",                # 모노
        "-ar", "16000",            # 16kHz (Whisper 최적)
        "-f", "wav",
        str(TEMP_WAV_PATH)
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print("[오류] 오디오 변환 실패:")
        print(result.stderr.decode(errors="ignore"))
        sys.exit(1)

    print("[변환 완료]")
    return TEMP_WAV_PATH


def load_audio_file(file_path_str: str) -> Path:
    """오디오 파일 로드 및 wav 변환"""
    input_path = Path(file_path_str)

    if not input_path.exists():
        print(f"[오류] 파일을 찾을 수 없습니다: {file_path_str}")
        sys.exit(1)

    if input_path.suffix.lower() not in SUPPORTED_FORMATS:
        print(f"[오류] 지원하지 않는 형식입니다. 지원 형식: {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)

    # wav 파일은 바로 사용, 나머지는 변환
    if input_path.suffix.lower() == ".wav":
        return input_path
    else:
        return convert_to_wav(input_path)


def record_microphone() -> Path:
    """마이크로 실시간 녹음 (Enter로 종료)"""
    sample_rate = 16000
    print("[마이크 녹음] 녹음을 시작합니다. 종료하려면 Enter를 누르세요...")

    recorded_chunks = []

    def callback(indata, frames, time, status):
        recorded_chunks.append(indata.copy())

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
        input()  # Enter 입력까지 대기

    print("[녹음 완료] 파일 저장 중...")
    audio_data = np.concatenate(recorded_chunks, axis=0)
    audio_int16 = (audio_data * 32767).astype(np.int16)
    wavfile.write(TEMP_WAV_PATH, sample_rate, audio_int16)
    return TEMP_WAV_PATH


# ============================================================
# STT + 화자 분리 통합 (whisperx)
# ============================================================

def run_whisperx(wav_path: Path, hf_token: str) -> list:
    """
    whisperx로 STT + 화자 분리를 한 번에 처리
    반환: [{"speaker": "SPEAKER_00", "text": "안녕하세요"}, ...]
    """
    try:
        import whisperx
    except ImportError:
        print("[오류] whisperx가 설치되지 않았습니다: pip install whisperx")
        sys.exit(1)

    # whisperx 내부에서 ffmpeg를 subprocess로 호출하므로 PATH에 추가
    ffmpeg_path = _get_ffmpeg_path()
    ffmpeg_dir = str(Path(ffmpeg_path).parent)
    os.environ["PATH"] = ffmpeg_dir + os.pathsep + os.environ.get("PATH", "")

    device = "cpu"  # GPU 없으면 cpu
    compute_type = "int8"

    # 오디오 길이 확인
    sample_rate, audio_data = wavfile.read(wav_path)
    total_duration = len(audio_data) / sample_rate
    print(f"[음성 변환 중] 총 길이: {total_duration / 60:.1f}분")

    # 1) 오디오 로드
    print("[음성 변환 중] 오디오 로딩 중...")
    audio = whisperx.load_audio(str(wav_path))

    # 2) STT - 한/영 혼용 (language=None 자동 감지)
    print("[음성 변환 중] Whisper large-v3-turbo 실행 중...")
    model = whisperx.load_model("large-v3-turbo", device=device, compute_type=compute_type)
    result = model.transcribe(audio, batch_size=8, language=None)
    print(f"[음성 변환 완료] 감지 언어: {result.get('language', '알 수 없음')}")

    # 3) 정밀 타임스탬프 정렬
    print("[타임스탬프 정렬 중]...")
    detected_lang = result.get("language", "ko")
    try:
        model_a, metadata = whisperx.load_align_model(language_code=detected_lang, device=device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, device)
    except Exception:
        # 정렬 모델 없는 언어는 건너뜀
        pass

    # 4) 화자 분리 (내부적으로 pyannote 사용, HF 토큰 필요)
    print("[화자 분리 중] whisperx diarization 실행 중...")
    from whisperx import diarize as wx_diarize
    diarize_model = wx_diarize.DiarizationPipeline(token=hf_token, device=device)
    diarize_segments = diarize_model(audio)

    # 5) 화자 정보를 각 세그먼트에 자동 매핑
    result = wx_diarize.assign_word_speakers(diarize_segments, result)

    # 6) 세그먼트 → {"speaker", "text"} 리스트로 변환 후 연속 화자 합치기
    raw = []
    for seg in result["segments"]:
        speaker = seg.get("speaker", "SPEAKER_00")
        text = seg.get("text", "").strip()
        if text:
            raw.append({"speaker": speaker, "text": text})

    # 같은 화자가 연속되면 합치기
    merged = []
    for item in raw:
        if merged and merged[-1]["speaker"] == item["speaker"]:
            merged[-1]["text"] += " " + item["text"]
        else:
            merged.append(item.copy())

    speaker_count = len(set(item["speaker"] for item in merged))
    print(f"[화자 분리 완료] 화자 {speaker_count}명 감지")
    return merged


# ============================================================
# 화자 이름 매핑 적용
# ============================================================

def apply_speaker_names(transcript: list, speaker_names: dict) -> list:
    """
    config.yaml의 speaker_names 매핑 적용
    매핑 없는 화자는 "Speaker 1", "Speaker 2" 형식으로 자동 부여
    """
    seen_speakers = {}
    counter = 1
    for item in transcript:
        sid = item["speaker"]
        if sid not in seen_speakers:
            if speaker_names.get(sid):
                seen_speakers[sid] = speaker_names[sid]
            else:
                seen_speakers[sid] = f"Speaker {counter}"
            counter += 1

    result = []
    for item in transcript:
        result.append({
            "speaker": seen_speakers[item["speaker"]],
            "text": item["text"]
        })
    return result


# ============================================================
# Ollama 요약 생성
# ============================================================

def build_prompt(transcript: list) -> str:
    """Ollama에 전달할 프롬프트 생성 (회의 내용은 로그에 출력하지 않음)"""
    script_lines = [f"{item['speaker']}: {item['text']}" for item in transcript]
    script_text = "\n".join(script_lines)

    prompt = f"""당신은 회의록 작성 전문가입니다. 아래 회의 스크립트를 분석하여 구조화된 회의록을 작성해주세요.

회의 스크립트:
{script_text}

다음 형식으로 작성해주세요. 각 섹션을 명확히 구분하고 한국어로 작성하세요:

[중요 내용 요약]
- 회의의 핵심 논의 사항을 3~7개 bullet point로 요약

[Technical Detail 요약]
- 기술적 내용이 있다면 요약, 없으면 "해당 없음"

[액션 아이템]
화자별로 구분하여 각 사람의 할 일 목록 작성
형식: 화자이름 | 액션아이템

출력은 위 세 섹션만 작성하세요."""

    return prompt


def run_ollama_summary(transcript: list, ollama_config: dict) -> dict:
    """
    Ollama API로 회의록 요약 생성
    반환: {"summary": "...", "technical": "...", "action_items": {...}}
    """
    print("[요약 생성 중] Ollama qwen2.5:14b 실행 중...")

    host = ollama_config.get("host", "http://localhost:11434")
    model = ollama_config.get("model", "qwen2.5:14b")
    timeout = ollama_config.get("timeout", 300)

    # Ollama 연결 확인
    try:
        health = requests.get(f"{host}/api/tags", timeout=10)
        health.raise_for_status()
    except requests.exceptions.ConnectionError:
        print(f"[오류] Ollama에 연결할 수 없습니다. Ollama가 실행 중인지 확인하세요: {host}")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("[오류] Ollama 연결 시간 초과")
        sys.exit(1)

    prompt = build_prompt(transcript)

    try:
        response = requests.post(
            f"{host}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=timeout
        )
        response.raise_for_status()
    except requests.exceptions.Timeout:
        print(f"[오류] Ollama 응답 시간 초과 ({timeout}초). config.yaml의 timeout 값을 늘려주세요.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[오류] Ollama API 오류: {e}")
        sys.exit(1)

    raw_output = response.json().get("response", "")
    return _parse_ollama_output(raw_output)


def _parse_ollama_output(raw: str) -> dict:
    """Ollama 출력에서 섹션별 내용 추출"""
    sections = {"summary": "", "technical": "", "action_items": {}}

    # 중요 내용 요약 추출
    if "[중요 내용 요약]" in raw:
        start = raw.index("[중요 내용 요약]") + len("[중요 내용 요약]")
        end = raw.index("[Technical Detail 요약]") if "[Technical Detail 요약]" in raw else len(raw)
        sections["summary"] = raw[start:end].strip()

    # Technical Detail 요약 추출
    if "[Technical Detail 요약]" in raw:
        start = raw.index("[Technical Detail 요약]") + len("[Technical Detail 요약]")
        end = raw.index("[액션 아이템]") if "[액션 아이템]" in raw else len(raw)
        sections["technical"] = raw[start:end].strip()

    # 액션 아이템 추출 및 화자별 분류
    if "[액션 아이템]" in raw:
        start = raw.index("[액션 아이템]") + len("[액션 아이템]")
        action_raw = raw[start:].strip()

        action_by_speaker = {}
        for line in action_raw.splitlines():
            line = line.strip()
            if "|" in line:
                parts = line.split("|", 1)
                speaker = parts[0].strip().lstrip("-").strip()
                action = parts[1].strip()
                if speaker not in action_by_speaker:
                    action_by_speaker[speaker] = []
                action_by_speaker[speaker].append(action)
        sections["action_items"] = action_by_speaker

    return sections


# ============================================================
# Obsidian 마크다운 생성 및 저장
# ============================================================

def build_markdown(date_str: str, summary_data: dict, transcript: list) -> str:
    """Obsidian 형식의 마크다운 회의록 생성"""
    lines = []
    lines.append(f"# 회의록 - {date_str}")
    lines.append("")

    lines.append("## 📋 중요 내용 요약")
    lines.append("")
    lines.append(summary_data.get("summary", ""))
    lines.append("")

    lines.append("## 🔧 Technical Detail 요약")
    lines.append("")
    lines.append(summary_data.get("technical", ""))
    lines.append("")

    lines.append("## ✅ 액션 아이템")
    lines.append("")
    action_items = summary_data.get("action_items", {})
    if action_items:
        for speaker, items in action_items.items():
            lines.append(f"### {speaker}")
            for item in items:
                lines.append(f"- [ ] {item}")
            lines.append("")
    else:
        lines.append("(액션 아이템을 파싱하지 못했습니다. 아래 스크립트를 참고하세요)")
        lines.append("")

    lines.append("## 📝 전체 스크립트")
    lines.append("")
    for item in transcript:
        lines.append(f"**{item['speaker']}**: {item['text']}")
        lines.append("")

    return "\n".join(lines)


def get_output_path(vault_path: str, date_str: str) -> Path:
    """
    저장 경로 결정 - 같은 날짜 파일 존재 시 자동 넘버링
    예: 2025-01-15_회의록.md → 2025-01-15_회의록_2.md
    """
    vault = Path(vault_path)
    vault.mkdir(parents=True, exist_ok=True)

    base_name = f"{date_str}_회의록"
    output_path = vault / f"{base_name}.md"

    if not output_path.exists():
        return output_path

    counter = 2
    while True:
        numbered_path = vault / f"{base_name}_{counter}.md"
        if not numbered_path.exists():
            return numbered_path
        counter += 1


def save_to_obsidian(markdown_content: str, output_path: Path) -> None:
    """Obsidian vault에 마크다운 파일 저장"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_content)
    print(f"[저장 완료] {output_path}")


# ============================================================
# 임시 파일 정리
# ============================================================

def cleanup_temp_files(wav_path: Path) -> None:
    """처리 완료 후 변환된 임시 wav 파일만 삭제 (원본은 보존)"""
    if wav_path == TEMP_WAV_PATH and TEMP_WAV_PATH.exists():
        TEMP_WAV_PATH.unlink()
        print("[정리 완료] 임시 파일 삭제됨")


# ============================================================
# 메인 실행
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="회의록 자동화 시스템")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", type=str, help="오디오 파일 경로 (mp3, wav, m4a, mp4)")
    group.add_argument("--mic", "-m", action="store_true", help="마이크 실시간 녹음")
    args = parser.parse_args()

    # 설정 로드
    config = load_config()
    hf_token = config["huggingface_token"]
    obsidian_path = config["obsidian_vault_path"]
    speaker_names = config.get("speaker_names", {}) or {}
    ollama_config = config.get("ollama", {})

    date_str = datetime.now().strftime("%Y-%m-%d")

    # 1단계: 오디오 입력
    if args.mic:
        wav_path = record_microphone()
    else:
        wav_path = load_audio_file(args.file)

    try:
        # 2단계: STT + 화자 분리 (whisperx 통합)
        transcript_raw = run_whisperx(wav_path, hf_token)

        # 3단계: 화자 이름 매핑
        transcript = apply_speaker_names(transcript_raw, speaker_names)

        # 5단계: Ollama 요약
        summary_data = run_ollama_summary(transcript, ollama_config)

        # 6단계: 마크다운 생성
        markdown_content = build_markdown(date_str, summary_data, transcript)

        # 7단계: Obsidian 저장
        output_path = get_output_path(obsidian_path, date_str)
        save_to_obsidian(markdown_content, output_path)

    finally:
        # 오류 발생 시에도 임시 파일 반드시 삭제
        cleanup_temp_files(wav_path)

    print(f"\n[완료] 회의록이 저장되었습니다: {output_path}")


if __name__ == "__main__":
    main()