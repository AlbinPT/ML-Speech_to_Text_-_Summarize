import streamlit as st
import whisper
import torch
import time
import tempfile
import os
import re
import nltk

from textblob import TextBlob
from nltk.corpus import stopwords
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# App settings

st.set_page_config(page_title="AI Speech to Text", layout="wide")

st.markdown("""
<style>
.main-title {font-size:38px;font-weight:700;margin-bottom:5px;}
.subtitle {font-size:17px;color:#9ca3af;margin-bottom:25px;}
.section-title {font-size:24px;font-weight:600;margin-top:25px;margin-bottom:15px;}
</style>
""", unsafe_allow_html=True)


# Setup NLTK

@st.cache_resource
def setup_nltk():
    nltk.download("stopwords", quiet=True)

setup_nltk()


# Load Whisper model

@st.cache_resource
def load_whisper_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = whisper.load_model("small", device=device)
    return model, device

model, device = load_whisper_model()


# Load summary model

@st.cache_resource
def load_summary_model():
    name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(name)
    summary_model = AutoModelForSeq2SeqLM.from_pretrained(name)
    return tokenizer, summary_model


# Language settings

LANGUAGE_CODES = {
    "Auto Detect": None,
    "Malayalam": "ml",
    "English": "en",
    "Hindi": "hi"
}


# Initialize session state

if "text" not in st.session_state:
    st.session_state.text = ""

if "processing_time" not in st.session_state:
    st.session_state.processing_time = 0

if "detected_language" not in st.session_state:
    st.session_state.detected_language = ""

if "summary" not in st.session_state:
    st.session_state.summary = ""


# Transcribe audio

def transcribe_audio(audio_path, selected_language):

    start_time = time.time()
    language_code = LANGUAGE_CODES[selected_language]

    options = {
        "task": "transcribe",
        "fp16": torch.cuda.is_available(),
        "temperature": 0,
        "condition_on_previous_text": False,
        "verbose": False
    }

    if language_code:
        options["language"] = language_code

    result = model.transcribe(audio_path, **options)

    processing_time = round(time.time() - start_time, 2)
    text = result["text"].strip()

    detected_language = result.get(
        "language",
        language_code or "Unknown"
    )

    return text, processing_time, detected_language


# Analyze text

def analyze_text(text):

    words = text.split()
    sentences = [s for s in re.split(r"[.!?]+", text) if s.strip()]

    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity
    subjectivity = analysis.sentiment.subjectivity

    if polarity > 0.05:
        sentiment = "Positive"
    elif polarity < -0.05:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    stop_words = set(stopwords.words("english"))

    clean_words = re.findall(
        r"\b[a-zA-Z]+\b",
        text.lower()
    )

    filtered_words = [
        word for word in clean_words
        if word not in stop_words and len(word) > 2
    ]

    keywords = Counter(filtered_words).most_common(5)

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "keywords": keywords
    }


# Split long text

def split_text(text, max_words=450):

    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:

        sentence_words = len(sentence.split())

        if current_words + sentence_words > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_words = 0

        current_chunk.append(sentence)
        current_words += sentence_words

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# Generate summary

def generate_summary(text, tokenizer, summary_model, max_tokens):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    summary_ids = summary_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_tokens,
        min_new_tokens=max(20, max_tokens // 3),
        num_beams=4,
        length_penalty=2.0,
        early_stopping=True
    )

    return tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )


# Create AI summary

def summarize_text(text, summary_length):

    if not text or len(text.split()) < 20:
        return "The transcription is too short to summarize."

    tokenizer, summary_model = load_summary_model()

    settings = {
        "Short": 70,
        "Medium": 130,
        "Detailed": 220
    }

    max_tokens = settings[summary_length]
    chunks = split_text(text)

    summaries = []

    for chunk in chunks:

        try:
            summary = generate_summary(
                chunk,
                tokenizer,
                summary_model,
                max_tokens
            )

            summaries.append(summary)

        except Exception:
            summaries.append(chunk)

    final_summary = " ".join(summaries)

    # Final summary for long text

    if len(chunks) > 1:

        try:
            final_summary = generate_summary(
                final_summary,
                tokenizer,
                summary_model,
                max_tokens
            )

        except Exception:
            pass

    return final_summary


# Save temporary audio

def save_temp_audio(audio_file, suffix):

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix
    ) as temp_file:

        temp_file.write(audio_file.getvalue())
        return temp_file.name


# Process audio

def process_audio(audio_file, suffix, selected_language):

    temp_path = None

    try:

        temp_path = save_temp_audio(audio_file, suffix)

        text, processing_time, detected_language = transcribe_audio(
            temp_path,
            selected_language
        )

        st.session_state.text = text
        st.session_state.processing_time = processing_time
        st.session_state.detected_language = detected_language
        st.session_state.summary = ""

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# Show results

def display_results():

    text = st.session_state.text
    results = analyze_text(text)

    st.divider()


    # Transcribed text

    st.header("Transcribed Text")

    st.text_area(
        "Result",
        value=text,
        height=200
    )

    st.download_button(
        "Download Transcribed Text",
        data=text,
        file_name="transcribed_text.txt",
        mime="text/plain"
    )


    # Processing information

    st.header("Processing Information")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Processing Time",
        f"{st.session_state.processing_time} seconds"
    )

    col2.metric(
        "Language",
        str(st.session_state.detected_language).upper()
    )

    col3.metric(
        "Words",
        results["word_count"]
    )

    col4.metric(
        "Sentences",
        results["sentence_count"]
    )


    # Text analysis

    st.header("Text Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Sentiment",
        results["sentiment"]
    )

    col2.metric(
        "Polarity",
        round(results["polarity"], 3)
    )

    col3.metric(
        "Subjectivity",
        round(results["subjectivity"], 3)
    )


    # Keywords

    if results["keywords"]:

        keyword_text = ", ".join(
            word for word, count in results["keywords"]
        )

        st.write("**Top Keywords:**", keyword_text)

    else:
        st.write("No significant keywords found.")


    # AI summarizer

    st.divider()

    st.header("AI Text Summarizer")

    summary_length = st.selectbox(
        "Summary Length",
        ["Short", "Medium", "Detailed"],
        index=1
    )

    if st.button("Generate Summary"):

        try:

            with st.spinner("Generating AI summary..."):

                st.session_state.summary = summarize_text(
                    text,
                    summary_length
                )

        except Exception as error:
            st.error(f"Summary Error: {error}")


    # Show summary

    if st.session_state.summary:

        st.subheader("Summarized Text")

        st.text_area(
            "Summary",
            value=st.session_state.summary,
            height=200
        )

        st.download_button(
            "Download Summary",
            data=st.session_state.summary,
            file_name="summarized_text.txt",
            mime="text/plain"
        )


# Main application

st.markdown(
    '<div class="main-title">AI Speech to Text</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Convert speech into text and analyze it using NLP and AI.'
    '</div>',
    unsafe_allow_html=True
)


# GPU status

if device == "cuda":

    st.success(
        f"GPU Enabled: {torch.cuda.get_device_name(0)}"
    )

else:
    st.warning("GPU is not available. Running on CPU.")


# Language selection

selected_language = st.selectbox(
    "Select the spoken language",
    list(LANGUAGE_CODES.keys())
)


# Audio input

st.header("Audio Input")

microphone_tab, upload_tab = st.tabs([
    "Record Microphone",
    "Upload Audio File"
])


# Microphone input



with microphone_tab:

    recorded_audio = st.audio_input("Record Audio")

    if recorded_audio is not None:

        st.audio(recorded_audio)

        if st.button(
            "Convert Microphone Audio",
            key="microphone_button"
        ):

            try:

                with st.spinner("Converting speech to text..."):

                    process_audio(
                        recorded_audio,
                        ".wav",
                        selected_language
                    )

                st.success("Transcription completed.")

            except Exception as error:
                st.error(f"Error processing audio: {error}")

# Upload audio

with upload_tab:

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "mp4", "mpeg", "webm"]
    )

    if uploaded_file is not None:

        st.audio(uploaded_file)

        if st.button("Convert Uploaded Audio", key="upload_button"):

            try:

                suffix = os.path.splitext(uploaded_file.name)[1]

                with st.spinner("Converting speech to text..."):

                    process_audio(
                        uploaded_file,
                        suffix,
                        selected_language
                    )

                st.success("Transcription completed.")

            except Exception as error:
                st.error(f"Error processing audio: {error}")

# Display results
if st.session_state.text:
    display_results()
