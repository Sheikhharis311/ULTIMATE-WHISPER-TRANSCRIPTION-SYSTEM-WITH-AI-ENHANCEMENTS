# ULTIMATE WHISPER TRANSCRIPTION SYSTEM WITH AI ENHANCEMENTS

A comprehensive, high-performance, and AI-augmented speech-to-text transcription system built on **OpenAI Whisper**. This system offers real-time audio transcription with intelligent AI text analysis, advanced audio enhancement, and multi-format export capabilities — optimized for professional-grade applications in research, media, accessibility services, and enterprise audio processing.

---

## PROJECT OVERVIEW

This system extends the capabilities of OpenAI’s Whisper by integrating:

- Advanced audio enhancement techniques (noise reduction and spectral enhancement)
- AI-driven text analysis modules for sentiment analysis, summarization, and keyword extraction
- Multi-format transcription outputs (TXT, JSON, SRT, VTT)
- Batch processing with detailed performance reporting
- Multi-GPU and distributed processing support

Built using **PyTorch** and state-of-the-art models from **Hugging Face Transformers** and **SpeechBrain**, this system delivers fast, reliable, and scalable transcription workflows suitable for production environments.

---

## FEATURES

- Accurate speech-to-text transcription powered by OpenAI Whisper
- Real-time and batch processing modes
- Advanced audio enhancement:
  - Spectral masking enhancement
  - Stationary noise reduction
- AI-powered text analysis:
  - Sentiment detection
  - Summarization of long transcripts
  - Keyword extraction for content indexing
- Support for multiple Whisper model sizes: `tiny`, `base`, `small`, `medium`, `large`
- Multi-format output support:
  - Plain text `.txt`
  - Structured metadata `.json`
  - Subtitle files `.srt`, `.vtt`
- Batch directory processing with cumulative report generation
- Configurable processing options: language, temperature, beam search, and more
- Full GPU acceleration with fallbacks for CPU and Apple MPS

---

## DEPENDENCIES

- Python 3.8+
- PyTorch (with optional CUDA support)
- OpenAI Whisper
- Hugging Face Transformers
- SpeechBrain
- noisereduce
- pydub
- soundfile
- scikit-learn
- tqdm
- matplotlib

### Install all dependencies:

```bash
pip install -r requirements.txt
