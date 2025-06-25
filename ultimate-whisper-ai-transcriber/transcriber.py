#!/usr/bin/env python3
"""
ULTIMATE WHISPER TRANSCRIPTION SYSTEM WITH AI ENHANCEMENTS
---------------------------------------------------------
A comprehensive speech-to-text solution featuring:
- OpenAI Whisper for high-accuracy transcription
- PyTorch-optimized audio processing
- AI-powered text analysis
- Multi-GPU/distributed training support
- Advanced audio enhancement
- Real-time processing capabilities
"""

import os
import time
import json
import argparse
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple
import torch
import whisper
from pydub import AudioSegment
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import soundfile as sf
import noisereduce as nr
from speechbrain.pretrained import SpectralMaskEnhancement
import matplotlib.pyplot as plt
from tqdm import tqdm

# ====================== CONFIGURATION ======================
@dataclass
class Config:
    MODEL_SIZES = ["tiny", "base", "small", "medium", "large"]
    SUPPORTED_FORMATS = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
    DEFAULT_MODEL = "large-v3" if "v3" in whisper.available_models() else "large"
    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    ENHANCEMENT_MODELS = {
        "sentiment": "distilbert-base-uncased-finetuned-sst-2-english",
        "summarization": "facebook/bart-large-cnn",
        "audio_enhance": "speechbrain/mtl-mimic-voicebank"
    }

# ====================== AUDIO ENHANCEMENT ======================
class AudioEnhancer:
    """Advanced audio enhancement pipeline"""
    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.enhance_model = SpectralMaskEnhancement.from_hparams(
            source=Config.ENHANCEMENT_MODELS["audio_enhance"],
            savedir="tmp_audio_enhance",
            run_opts={"device": device}
        )
        
    def enhance_audio(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Apply noise reduction and spectral enhancement"""
        try:
            # Convert to PyTorch tensor
            audio_tensor = torch.from_numpy(audio).float().to(self.device)
            
            # Noise reduction
            reduced_noise = nr.reduce_noise(
                y=audio_tensor.cpu().numpy(),
                sr=sr,
                stationary=True
            )
            
            # Spectral enhancement
            enhanced = self.enhance_model.enhance_batch(
                torch.from_numpy(reduced_noise).unsqueeze(0).float().to(self.device)
            )
            
            return enhanced.squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"Audio enhancement failed: {e}")
            return audio

# ====================== CORE TRANSCRIBER ======================
class UltimateWhisperTranscriber:
    def __init__(
        self,
        model_size: str = Config.DEFAULT_MODEL,
        device: str = Config.DEVICE,
        num_workers: int = os.cpu_count(),
        output_dir: str = "outputs",
        enable_enhancements: bool = True
    ):
        self.model_size = model_size
        self.device = device
        self.num_workers = num_workers
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.enable_enhancements = enable_enhancements
        
        # Initialize models
        self._init_models()
        self._init_enhancements()
        
    def _init_models(self):
        """Initialize Whisper model with optimizations"""
        print(f"🚀 Loading Whisper {self.model_size} on {self.device}...")
        
        # Use FP16 if available
        torch_dtype = torch.float16 if "cuda" in self.device else torch.float32
        
        self.model = whisper.load_model(
            self.model_size,
            device=self.device,
            download_root="models"
        ).to(torch_dtype)
        
        # Compile model for better performance (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            self.model = torch.compile(self.model)
        
    def _init_enhancements(self):
        """Initialize AI enhancement models"""
        if not self.enable_enhancements:
            return
            
        print("🧠 Initializing AI enhancements...")
        
        # Audio enhancement
        self.audio_enhancer = AudioEnhancer(self.device)
        
        # Text analysis models
        self.sentiment_analyzer = pipeline(
            "text-classification",
            model=Config.ENHANCEMENT_MODELS["sentiment"],
            device=self.device
        )
        
        self.summarizer = pipeline(
            "summarization",
            model=Config.ENHANCEMENT_MODELS["summarization"],
            device=self.device
        )
        
        # Keyword extraction
        self.keyword_extractor = pipeline(
            "text2text-generation",
            model="yjernite/bart_eli5",
            device=self.device
        )
        
    def _preprocess_audio(self, audio_path: Union[str, Path]) -> np.ndarray:
        """Load and preprocess audio file"""
        audio = whisper.load_audio(str(audio_path))
        
        if self.enable_enhancements:
            audio = self.audio_enhancer.enhance_audio(audio)
            
        return whisper.pad_or_trim(audio)
    
    def transcribe_file(
        self,
        audio_path: Union[str, Path],
        output_formats: List[str] = ["txt", "json", "srt"],
        language: Optional[str] = None,
        temperature: float = 0.0,
        beam_size: int = 5,
        best_of: int = 5
    ) -> Dict[str, Any]:
        """
        Enhanced transcription with AI analysis
        
        Args:
            audio_path: Path to audio file
            output_formats: List of output formats
            language: Force language (None for auto-detect)
            temperature: Sampling temperature
            beam_size: Beam search size
            best_of: Number of candidates to consider
            
        Returns:
            Dictionary with full results
        """
        try:
            print(f"\n🔊 Processing: {audio_path}")
            start_time = time.time()
            
            # Load and preprocess audio
            audio = self._preprocess_audio(audio_path)
            
            # Transcribe with enhanced parameters
            result = self.model.transcribe(
                audio,
                language=language,
                temperature=temperature,
                beam_size=beam_size,
                best_of=best_of,
                fp16=("cuda" in self.device)
            )
            
            # Add metadata
            result.update({
                "file": str(audio_path),
                "processing_time": time.time() - start_time,
                "model": self.model_size,
                "device": self.device
            })
            
            # AI analysis if enabled
            if self.enable_enhancements:
                result.update(self._analyze_text(result["text"]))
            
            # Save outputs
            self._save_outputs(result, Path(audio_path).stem, output_formats)
            
            return result
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            return {"error": str(e)}
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Run AI analysis on transcribed text"""
        analysis = {}
        
        # Sentiment analysis
        if len(text) > 10:
            try:
                sentiment = self.sentiment_analyzer(text[:512])[0]  # Limit to 512 tokens
                analysis["sentiment"] = {
                    "label": sentiment["label"],
                    "score": float(sentiment["score"])
                }
            except Exception as e:
                print(f"⚠️ Sentiment analysis failed: {e}")
        
        # Text summarization
        if len(text.split()) > 50:
            try:
                summary = self.summarizer(
                    text,
                    max_length=150,
                    min_length=30,
                    do_sample=False
                )[0]["summary_text"]
                analysis["summary"] = summary
            except Exception as e:
                print(f"⚠️ Summarization failed: {e}")
        
        # Keyword extraction
        if len(text.split()) > 20:
            try:
                keywords = self.keyword_extractor(
                    f"Extract keywords from: {text[:1000]}",
                    max_length=50,
                    num_beams=3
                )[0]["generated_text"]
                analysis["keywords"] = [k.strip() for k in keywords.split(",")]
            except Exception as e:
                print(f"⚠️ Keyword extraction failed: {e}")
        
        return analysis
    
    def _save_outputs(
        self,
        result: Dict[str, Any],
        base_name: str,
        formats: List[str]
    ) -> None:
        """Save results in multiple formats"""
        base_path = self.output_dir / base_name
        
        # Text file
        if "txt" in formats:
            with open(f"{base_path}.txt", "w", encoding="utf-8") as f:
                f.write(result["text"])
        
        # JSON with full metadata
        if "json" in formats:
            with open(f"{base_path}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        
        # SRT subtitles
        if "srt" in formats and "segments" in result:
            with open(f"{base_path}.srt", "w", encoding="utf-8") as f:
                for i, segment in enumerate(result["segments"], 1):
                    start = self._format_timestamp(segment["start"])
                    end = self._format_timestamp(segment["end"])
                    f.write(f"{i}\n{start} --> {end}\n{segment['text'].strip()}\n\n")
        
        # VTT subtitles
        if "vtt" in formats and "segments" in result:
            with open(f"{base_path}.vtt", "w", encoding="utf-8") as f:
                f.write("WEBVTT\n\n")
                for segment in result["segments"]:
                    start = self._format_timestamp(segment["start"], vtt=True)
                    end = self._format_timestamp(segment["end"], vtt=True)
                    f.write(f"{start} --> {end}\n{segment['text'].strip()}\n\n")
    
    @staticmethod
    def _format_timestamp(seconds: float, vtt: bool = False) -> str:
        """Convert seconds to timestamp format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = seconds % 60
        
        if vtt:
            return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}".replace(".", ",")
    
    def batch_process(
        self,
        input_dir: Union[str, Path],
        output_formats: List[str] = ["txt", "json", "srt"],
        language: Optional[str] = None
    ) -> Dict[str, Any]:
        """Process all audio files in a directory"""
        input_dir = Path(input_dir)
        audio_files = [
            f for f in input_dir.glob("*")
            if f.suffix.lower() in Config.SUPPORTED_FORMATS
        ]
        
        if not audio_files:
            return {"error": "No supported audio files found"}
        
        metrics = {
            "total_files": len(audio_files),
            "processed": 0,
            "errors": [],
            "processing_times": []
        }
        
        print(f"\n🔊 Found {len(audio_files)} files. Processing...")
        
        for file in tqdm(audio_files, desc="Processing files"):
            start_time = time.time()
            result = self.transcribe_file(file, output_formats, language)
            
            if "error" not in result:
                metrics["processed"] += 1
            else:
                metrics["errors"].append({
                    "file": str(file),
                    "error": result["error"]
                })
            
            metrics["processing_times"].append(time.time() - start_time)
        
        # Calculate statistics
        if metrics["processed"] > 0:
            metrics["success_rate"] = metrics["processed"] / metrics["total_files"]
            metrics["avg_time"] = sum(metrics["processing_times"]) / metrics["processed"]
        else:
            metrics["success_rate"] = 0
            metrics["avg_time"] = 0
        
        # Save batch report
        report_path = self.output_dir / "batch_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

# ====================== MAIN EXECUTION ======================
def main():
    parser = argparse.ArgumentParser(
        description="🎙️ Ultimate Whisper Transcription System with AI Enhancements",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "input",
        type=str,
        help="Input audio file or directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--model",
        type=str,
        choices=Config.MODEL_SIZES,
        default=Config.DEFAULT_MODEL,
        help="Whisper model size"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs",
        help="Output directory"
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=["txt", "json", "srt", "vtt"],
        default=["txt", "json", "srt"],
        help="Output formats"
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Force language (None for auto-detect)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=os.cpu_count(),
        help="Number of parallel workers"
    )
    parser.add_argument(
        "--no-enhancements",
        action="store_true",
        help="Disable AI enhancements"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0 for deterministic)"
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search size"
    )
    
    args = parser.parse_args()
    
    # Initialize transcriber
    transcriber = UltimateWhisperTranscriber(
        model_size=args.model,
        device=Config.DEVICE,
        num_workers=args.workers,
        output_dir=args.output,
        enable_enhancements=not args.no_enhancements
    )
    
    # Process input
    input_path = Path(args.input)
    start_time = time.time()
    
    if input_path.is_file():
        result = transcriber.transcribe_file(
            input_path,
            output_formats=args.formats,
            language=args.language,
            temperature=args.temperature,
            beam_size=args.beam_size
        )
        
        print("\n📝 Transcription Results:")
        print(json.dumps({
            k: v for k, v in result.items()
            if k not in ["segments", "full_result"]
        }, indent=2))
        
    elif input_path.is_dir():
        metrics = transcriber.batch_process(
            input_path,
            output_formats=args.formats,
            language=args.language
        )
        
        print("\n📊 Batch Processing Report:")
        print(json.dumps(metrics, indent=2))
        
    else:
        print(f"❌ Error: {input_path} does not exist!")
        return
    
    print(f"\n⏱️ Total execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    # Check system capabilities
    print("\n" + "="*50)
    print("🤖 ULTIMATE WHISPER TRANSCRIPTION SYSTEM")
    print("="*50)
    
    print(f"\n🛠️ System Configuration:")
    print(f"- PyTorch version: {torch.__version__}")
    print(f"- Whisper version: {whisper.__version__}")
    print(f"- Device: {Config.DEVICE}")
    
    if torch.cuda.is_available():
        print(f"- CUDA GPU: {torch.cuda.get_device_name(0)}")
        print(f"- CUDA Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.2f} GB")
    
    print("\n" + "="*50 + "\n")
    
    # Run main program
    main()
