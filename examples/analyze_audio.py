#!/usr/bin/env python3
"""
Example script to analyze a sleep recording using NeendAI.

Usage:
    python examples/analyze_audio.py path/to/recording.wav
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.audio import AudioPreprocessor, FeatureExtractor
from src.ml import OllamaClassifier, AHICalculator


def analyze_recording(file_path: str):
    """
    Analyze a sleep recording and generate AHI report.
    """
    print(f"Analyzing: {file_path}")
    print("=" * 50)

    # Initialize components
    preprocessor = AudioPreprocessor(sample_rate=16000)
    feature_extractor = FeatureExtractor(sample_rate=16000)
    classifier = OllamaClassifier(model="llama3.2")
    ahi_calculator = AHICalculator(segment_duration=30.0)

    # Load audio
    print("Loading audio...")
    audio = preprocessor.load_audio(file_path)
    duration_seconds = len(audio) / preprocessor.sr
    print(f"Duration: {duration_seconds:.1f} seconds ({duration_seconds/3600:.2f} hours)")

    # Segment audio
    segments = preprocessor.segment_audio(audio, segment_duration=30.0)
    print(f"Total segments: {len(segments)}")
    print()

    # Classify each segment
    events_timeline = []
    print("Classifying segments...")

    for i, segment in enumerate(segments):
        # Preprocess
        processed = preprocessor.process_audio_segment(segment)

        # Extract features
        features = feature_extractor.extract_all_features(processed)

        # Classify
        result = classifier.classify(features)
        events_timeline.append(result['event_class'])

        # Progress indicator
        event_symbols = ['·', 'z', '~', '!']
        print(event_symbols[result['event_class']], end='', flush=True)

        if (i + 1) % 60 == 0:
            print(f" [{i+1}/{len(segments)}]")

    print()
    print()

    # Generate report
    report = ahi_calculator.generate_report(events_timeline, "Patient")
    print(report)

    # Hourly breakdown
    print("\nEvent Legend: · = Normal, z = Snoring, ~ = Hypopnea, ! = Apnea")


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio.py <audio_file>")
        print("\nSupported formats: WAV, MP3, FLAC, OGG")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    analyze_recording(file_path)


if __name__ == "__main__":
    main()
