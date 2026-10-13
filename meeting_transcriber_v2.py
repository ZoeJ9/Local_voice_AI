# ============================================================
# 회의록 자동화 시스템 v2
# pyannote (화자 분리) + SenseVoiceSmall (STT + 감정/이벤트 태그) + Ollama (qwen3:14b)
# 완전 로컬 실행 - 인터넷 불필요 (모델 최초 다운로드 제외)
# ============================================================

import argparse
import os
import re
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
# 화자 분리 (pyannote) + STT (SenseVoiceSmall)
# ============================================================

def diarize_speakers(wav_path: Path, hf_token: str) -> list:
    """
    pyannote로 화자 분리만 수행 (STT 없음)
    반환: [{"speaker": "SPEAKER_00", "start": 0.0, "end": 3.2}, ...] (연속 동일 화자 구간 병합됨)
    """
    from pyannote.audio import Pipeline

    print("[화자 분리 중] pyannote diarization 실행 중...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diarization = pipeline(str(wav_path))

    raw = [{"speaker": spk, "start": turn.start, "end": turn.end}
           for turn, _, spk in diarization.itertracks(yield_label=True)]

    # 연속된 동일 화자 구간 병합
    merged = []
    for item in raw:
        if merged and merged[-1]["speaker"] == item["speaker"]:
            merged[-1]["end"] = item["end"]
        else:
            merged.append(item.copy())

    speaker_count = len(set(item["speaker"] for item in merged))
    print(f"[화자 분리 완료] 화자 {speaker_count}명 감지, 구간 {len(merged)}개")
    return merged


_EMOTION_EVENT_RE = re.compile(r"<\|([A-Za-z_]+)\|>")
_IGNORED_TAGS = {"NEUTRAL", "Speech"}


def transcribe_segments(wav_path: Path, diarized_segments: list) -> list:
    """
    화자 분리된 구간별로 SenseVoice STT 실행
    반환: [{"speaker": "SPEAKER_00", "text": "...", "emotion_tags": ["Laughter"]}, ...]
    """
    try:
        from funasr import AutoModel
    except ImportError:
        print("[오류] funasr가 설치되지 않았습니다: pip install funasr modelscope")
        sys.exit(1)

    print("[음성 변환 중] SenseVoiceSmall 모델 로딩 중...")
    model = AutoModel(model="iic/SenseVoiceSmall", device="cpu", disable_update=True)

    sample_rate, audio_data = wavfile.read(wav_path)

    print(f"[음성 변환 중] 구간 {len(diarized_segments)}개 STT 실행 중...")
    results = []
    for seg in diarized_segments:
        start_sample = int(seg["start"] * sample_rate)
        end_sample = int(seg["end"] * sample_rate)
        clip = audio_data[start_sample:end_sample].astype(np.float32) / 32768.0
        if len(clip) < sample_rate * 0.2:  # 0.2초 미만 구간은 스킵
            continue

        res = model.generate(input=clip, language="auto", use_itn=True)
        raw_text = res[0]["text"]
        tags = [t for t in _EMOTION_EVENT_RE.findall(raw_text) if t not in _IGNORED_TAGS]
        clean_text = _EMOTION_EVENT_RE.sub("", raw_text).strip()

        if clean_text:
            results.append({"speaker": seg["speaker"], "text": clean_text, "emotion_tags": tags})

    print(f"[음성 변환 완료] 발화 {len(results)}건")
    return results


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
            "text": item["text"],
            "emotion_tags": item.get("emotion_tags", []),
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

[회의 전체 맥락]
이 회의의 전반적인 목적, 배경, 핵심 주제를 2~3문장으로 서술

[중요 내용 요약]
- 회의의 핵심 논의 사항을 3~7개 bullet point로 요약

[Technical Detail 요약]
- 기술적 내용이 있다면 요약, 없으면 "해당 없음"

[액션 아이템]
화자별로 구분하여 각 사람의 할 일 목록 작성
형식: 화자이름 | 액션아이템

[화자 태도 분석]
각 화자의 발언 전체를 분석하여 긍정/부정/중립 비율을 추정
형식: 화자이름 | 긍정 X% | 부정 X% | 중립 X% | 한줄 특징 설명

[발표자별 받은 코멘트]
각 발표자가 다른 참여자들로부터 받은 질문, 피드백, 지적 사항을 정리
형식: 발표자이름 | 코멘트 내용 | 코멘트한 사람

[발표자 영어 발표 피드백]
각 발표자의 영어 발표 실력과 전달 명확도를 분석. 실제 발언 내용을 근거로 평가.
형식:
발표자이름
- 영어 유창성: (상/중/하) - 근거
- 발음/억양: 특이사항 또는 "특이사항 없음"
- 문장 구성: 문법적 정확도와 어휘 수준 평가
- 전달 명확도: 논리적 흐름, 핵심 전달 여부 평가
- 개선 제안: 구체적인 개선점 1~2가지

출력은 위 일곱 섹션만 작성하세요."""

    return prompt


def run_ollama_summary(transcript: list, ollama_config: dict) -> dict:
    """
    Ollama API로 회의록 요약 생성
    반환: {"summary": "...", "technical": "...", "action_items": {...}}
    """
    host = ollama_config.get("host", "http://localhost:11434")
    model = ollama_config.get("model", "qwen3:14b")
    timeout = ollama_config.get("timeout", 300)

    print(f"[요약 생성 중] Ollama {model} 실행 중...")

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
            json={"model": model, "prompt": prompt, "stream": False, "think": False},
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


def _extract_section(raw: str, section_tag: str, next_tags: list) -> str:
    """raw 텍스트에서 section_tag ~ 다음 섹션 사이의 내용 추출"""
    if section_tag not in raw:
        return ""
    start = raw.index(section_tag) + len(section_tag)
    end = len(raw)
    for tag in next_tags:
        if tag in raw:
            end = min(end, raw.index(tag))
    return raw[start:end].strip()


def _parse_ollama_output(raw: str) -> dict:
    """Ollama 출력에서 섹션별 내용 추출"""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    all_tags = ["[회의 전체 맥락]", "[중요 내용 요약]", "[Technical Detail 요약]",
                "[액션 아이템]", "[화자 태도 분석]", "[발표자별 받은 코멘트]", "[발표자 영어 발표 피드백]"]
    sections = {"context": "", "summary": "", "technical": "", "action_items": {},
                "sentiment": {}, "presenter_comments": {}, "english_feedback": {}}

    def next_tags(current_tag):
        idx = all_tags.index(current_tag)
        return all_tags[idx + 1:]

    # 회의 전체 맥락
    sections["context"] = _extract_section(raw, "[회의 전체 맥락]", next_tags("[회의 전체 맥락]"))

    # 중요 내용 요약
    sections["summary"] = _extract_section(raw, "[중요 내용 요약]", next_tags("[중요 내용 요약]"))

    # Technical Detail 요약
    sections["technical"] = _extract_section(raw, "[Technical Detail 요약]", next_tags("[Technical Detail 요약]"))

    # 액션 아이템 - "화자이름 | 액션아이템" 형식 파싱
    if "[액션 아이템]" in raw:
        action_raw = _extract_section(raw, "[액션 아이템]", next_tags("[액션 아이템]"))
        action_by_speaker = {}
        for line in action_raw.splitlines():
            line = line.strip().lstrip("-").strip()
            if "|" in line:
                parts = line.split("|", 1)
                speaker = parts[0].strip()
                action = parts[1].strip()
                if speaker:
                    action_by_speaker.setdefault(speaker, []).append(action)
        sections["action_items"] = action_by_speaker

    # 화자 태도 분석 - "화자이름 | 긍정 X% | 부정 X% | 중립 X% | 설명" 형식 파싱
    if "[화자 태도 분석]" in raw:
        sentiment_raw = _extract_section(raw, "[화자 태도 분석]", next_tags("[화자 태도 분석]"))
        sentiment_by_speaker = {}
        for line in sentiment_raw.splitlines():
            line = line.strip().lstrip("-").strip()
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 4:
                speaker = parts[0]
                if speaker:
                    sentiment_by_speaker[speaker] = {
                        "positive": parts[1] if len(parts) > 1 else "",
                        "negative": parts[2] if len(parts) > 2 else "",
                        "neutral":  parts[3] if len(parts) > 3 else "",
                        "note":     parts[4] if len(parts) > 4 else "",
                    }
        sections["sentiment"] = sentiment_by_speaker

    # 발표자별 받은 코멘트 - "발표자이름 | 코멘트 내용 | 코멘트한 사람" 형식 파싱
    if "[발표자별 받은 코멘트]" in raw:
        comments_raw = _extract_section(raw, "[발표자별 받은 코멘트]", next_tags("[발표자별 받은 코멘트]"))
        comments_by_presenter = {}
        for line in comments_raw.splitlines():
            line = line.strip().lstrip("-").strip()
            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 2:
                    presenter = parts[0]
                    comment = parts[1] if len(parts) > 1 else ""
                    commenter = parts[2] if len(parts) > 2 else ""
                    if presenter:
                        comments_by_presenter.setdefault(presenter, []).append(
                            {"comment": comment, "from": commenter}
                        )
        sections["presenter_comments"] = comments_by_presenter

    # 발표자 영어 발표 피드백 - 발표자이름 + 하위 항목 블록 파싱
    if "[발표자 영어 발표 피드백]" in raw:
        feedback_raw = _extract_section(raw, "[발표자 영어 발표 피드백]", next_tags("[발표자 영어 발표 피드백]"))
        feedback_by_presenter = {}
        current_presenter = None
        current_block = []
        for line in feedback_raw.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            # 들여쓰기 없이 "-"로 시작하지 않는 줄을 발표자 이름으로 인식
            if not stripped.startswith("-") and not stripped.startswith("•"):
                if current_presenter and current_block:
                    feedback_by_presenter[current_presenter] = "\n".join(current_block)
                current_presenter = stripped
                current_block = []
            else:
                if current_presenter:
                    current_block.append(stripped)
        if current_presenter and current_block:
            feedback_by_presenter[current_presenter] = "\n".join(current_block)
        sections["english_feedback"] = feedback_by_presenter

    return sections


# ============================================================
# Obsidian 마크다운 생성 및 저장
# ============================================================

def build_markdown(date_str: str, summary_data: dict, transcript: list) -> str:
    """Obsidian 형식의 마크다운 회의록 생성"""
    lines = []
    lines.append(f"# 회의록 - {date_str}")
    lines.append("")

    # 회의 전체 맥락
    lines.append("## 🗂️ 회의 전체 맥락")
    lines.append("")
    lines.append(summary_data.get("context", ""))
    lines.append("")

    lines.append("## 📋 중요 내용 요약")
    lines.append("")
    lines.append(summary_data.get("summary", ""))
    lines.append("")

    lines.append("## 🔧 Technical Detail 요약")
    lines.append("")
    lines.append(summary_data.get("technical", ""))
    lines.append("")

    # 화자 태도 분석
    lines.append("## 🎭 화자 태도 분석")
    lines.append("")
    sentiment = summary_data.get("sentiment", {})
    if sentiment:
        lines.append("| 화자 | 긍정 | 부정 | 중립 | 특징 |")
        lines.append("|------|------|------|------|------|")
        for speaker, data in sentiment.items():
            lines.append(f"| {speaker} | {data.get('positive','')} | {data.get('negative','')} | {data.get('neutral','')} | {data.get('note','')} |")
    else:
        lines.append("(태도 분석 데이터 없음)")
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

    lines.append("## 💬 발표자별 받은 코멘트")
    lines.append("")
    presenter_comments = summary_data.get("presenter_comments", {})
    if presenter_comments:
        for presenter, comments in presenter_comments.items():
            lines.append(f"### {presenter}")
            for c in comments:
                from_str = f" _(by {c['from']})_" if c.get("from") else ""
                lines.append(f"- {c['comment']}{from_str}")
            lines.append("")
    else:
        lines.append("(코멘트 데이터 없음)")
        lines.append("")

    lines.append("## 🗣️ 발표자 영어 발표 피드백")
    lines.append("")
    english_feedback = summary_data.get("english_feedback", {})
    if english_feedback:
        for presenter, feedback in english_feedback.items():
            lines.append(f"### {presenter}")
            lines.append(feedback)
            lines.append("")
    else:
        lines.append("(영어 발표 피드백 없음)")
        lines.append("")

    lines.append("## 📝 전체 스크립트")
    lines.append("")
    for item in transcript:
        tags = item.get("emotion_tags", [])
        nuance = f" `[{', '.join(tags)}]`" if tags else ""
        lines.append(f"**{item['speaker']}**: {item['text']}{nuance}")
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
    parser = argparse.ArgumentParser(description="회의록 자동화 시스템 v2")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", type=str, help="오디오 파일 경로 (mp3, wav, m4a, mp4)")
    group.add_argument("--mic", "-m", action="store_true", help="마이크 실시간 녹음")
    args = parser.parse_args()

    # 설정 로드
    config = load_config()
    hf_token = config["huggingface_token"]
    obsidian_path = config["obsidian_vault_path"]
    speaker_names = config.get("speaker_names", {}) or {}
    ollama_config = config.get("ollama_v2", {})

    date_str = datetime.now().strftime("%Y-%m-%d")

    # 1단계: 오디오 입력
    if args.mic:
        wav_path = record_microphone()
    else:
        wav_path = load_audio_file(args.file)

    try:
        # 2단계: 화자 분리 (pyannote) → 구간별 STT (SenseVoiceSmall)
        diarized = diarize_speakers(wav_path, hf_token)
        transcript_raw = transcribe_segments(wav_path, diarized)

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