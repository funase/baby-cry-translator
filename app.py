import json
import io
import wave
import struct
import random

import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
st.set_page_config(
    page_title="Baby Cry Translator 👶",
    page_icon="👶",
    layout="centered",
)

# 無料枠で音声解析が可能なモデル
MODEL_CANDIDATES = ["gemini-2.5-flash", "gemini-flash-latest"]

CATEGORIES = {
    "hunger": "🍼 空腹",
    "sleepy": "😴 眠い",
    "burp": "💨 げっぷが必要（不快感）",
    "diaper": "🧷 おむつ/お腹の不快感",
    "other": "❓ 一般的な不快感/その他",
}

SYSTEM_PROMPT = """
あなたは小児科医および赤ちゃんの泣き声解析の専門家です。
送られた音声ファイルの周波数・トーン・リズム・泣き方（音程の変化、間隔、強弱、
長さ、喉の絞まり具合など）を分析し、泣いている理由を以下の5カテゴリについて
確率（%）で算出してください。確率の合計は必ず100になるようにしてください。

カテゴリ:
1. hunger      : 空腹
2. sleepy      : 眠い
3. burp        : げっぷが必要（不快感）
4. diaper      : おむつ/お腹の不快感
5. other       : 一般的な不快感/その他

あわせて、最も確率が高い原因に対する具体的な対処法のアドバイスを
日本語100〜200文字で生成してください。

出力は必ず以下のJSONスキーマに従ってください。余計な文章は一切出力しないでください。
{
  "probabilities": {
    "hunger": <数値>,
    "sleepy": <数値>,
    "burp": <数値>,
    "diaper": <数値>,
    "other": <数値>
  },
  "top_cause": "<最も確率が高いカテゴリのキー>",
  "advice": "<対処法アドバイス（100〜200文字）>"
}
"""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "probabilities": {
            "type": "object",
            "properties": {
                "hunger": {"type": "number"},
                "sleepy": {"type": "number"},
                "burp": {"type": "number"},
                "diaper": {"type": "number"},
                "other": {"type": "number"},
            },
            "required": ["hunger", "sleepy", "burp", "diaper", "other"],
        },
        "top_cause": {"type": "string"},
        "advice": {"type": "string"},
    },
    "required": ["probabilities", "top_cause", "advice"],
}


# ---------------------------------------------------------
# Gemini API 呼び出し
# ---------------------------------------------------------
def analyze_cry(audio_bytes: bytes, mime_type: str) -> dict:
    """音声バイト列をGeminiに送信し、JSON解析結果を返す"""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error("APIキーが設定されていません。Secrets に GEMINI_API_KEY を設定してください。")
        st.stop()

    client = genai.Client(api_key=api_key)

    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type=mime_type)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        temperature=0.3,
    )

    last_error = None
    for model in MODEL_CANDIDATES:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[audio_part, "この赤ちゃんの泣き声を解析してください。"],
                config=config,
            )
            return json.loads(response.text)
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"すべてのモデルで解析に失敗しました: {last_error}")


# ---------------------------------------------------------
# ホワイトノイズ生成（PythonだけでWAVを合成・外部ファイル不要）
# ---------------------------------------------------------
@st.cache_data
def make_white_noise_wav(seconds: int = 15, sample_rate: int = 22050) -> bytes:
    """ソフトなホワイトノイズのWAVバイト列を生成"""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        frames = bytearray()
        fade_len = int(sample_rate * 0.5)
        total = seconds * sample_rate
        prev = 0.0
        for i in range(total):
            white = random.uniform(-1.0, 1.0)
            prev = 0.98 * prev + 0.02 * white
            sample = prev * 0.5
            if i < fade_len:
                sample *= i / fade_len
            elif i > total - fade_len:
                sample *= (total - i) / fade_len
            frames += struct.pack("<h", int(sample * 32767))
        wf.writeframes(bytes(frames))
    return buf.getvalue()


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("👶 Baby Cry Translator")
st.caption("赤ちゃんの泣き声をAIが解析し、原因の確率と対処法を表示します（個人利用向け）")

with st.expander("⚠️ 免責事項（必ずお読みください）"):
    st.warning(
        "本アプリはAIによる参考情報であり、医学的診断ではありません。"
        "赤ちゃんの様子に異常を感じた場合は、必ず小児科医にご相談ください。"
    )

# --- 音声入力 ---
st.subheader("🎤 音声入力")
tab_rec, tab_upload = st.tabs(["マイクで録音（5秒程度）", "ファイルをアップロード"])

audio_bytes, mime_type = None, None

with tab_rec:
    st.info("録音ボタンを押して5秒ほど泣き声を録音し、停止してください。")
    recorded = st.audio_input("マイク録音")
    if recorded:
        audio_bytes = recorded.getvalue()
        mime_type = "audio/wav"
        st.audio(audio_bytes, format="audio/wav")

with tab_upload:
    uploaded = st.file_uploader(
        "音声ファイル（wav / mp3 / m4a / ogg / flac）",
        type=["wav", "mp3", "m4a", "ogg", "flac"],
    )
    if uploaded:
        mime_map = {
            "wav": "audio/wav", "mp3": "audio/mp3", "m4a": "audio/mp4",
            "ogg": "audio/ogg", "flac": "audio/flac",
        }
        ext = uploaded.name.rsplit(".", 1)[-1].lower()
        audio_bytes = uploaded.read()
        mime_type = mime_map.get(ext, "audio/wav")
        st.audio(audio_bytes)

# --- 解析実行 ---
st.subheader("🔍 解析")
if st.button("解析する", type="primary", use_container_width=True):
    if not audio_bytes:
        st.error("先に音声を録音またはアップロードしてください。")
    else:
        with st.spinner("Gemini AIが解析中です…（10〜20秒程度）"):
            try:
                result = analyze_cry(audio_bytes, mime_type)
                st.session_state["result"] = result
            except Exception as e:
                st.error(f"解析に失敗しました: {e}")

# --- 結果表示 ---
if "result" in st.session_state:
    result = st.session_state["result"]
    probs = result.get("probabilities", {})
    total = sum(probs.values()) or 1

    st.subheader("📊 解析結果（原因の確率）")
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    for key, val in sorted_probs:
        pct = val / total * 100
        st.markdown(f"**{CATEGORIES.get(key, key)} — {pct:.1f}%**")
        st.progress(min(pct / 100, 1.0))

    top = result.get("top_cause", sorted_probs[0][0])
    st.subheader(f"💡 対処法アドバイス（最有力：{CATEGORIES.get(top, top)}）")
    st.success(result.get("advice", "アドバイスを取得できませんでした。"))

# --- 泣き止め音 ---
st.subheader("🎵 泣き止めサポート音")
st.caption("子宮内の音に似たホワイトノイズです。小さめの音量で再生してください。")
noise_wav = make_white_noise_wav()
st.audio(noise_wav, format="audio/wav", loop=True)
