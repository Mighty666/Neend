"""
Real-time Audio Processing Pipeline with Streaming ML
"""

import numpy as np
import asyncio
from typing import Dict, Optional, AsyncGenerator
from collections import deque
import logging

from src.audio import AudioPreprocessor, FeatureExtractor
from src.ml.enhanced_classifier import EnhancedClassifier
from src.ml.online import StreamingAHI

logger = logging.getLogger(__name__)


class RealTimeProcessor:
    """
    Real-time audio processing with streaming classification and AHI calculation.
    Optimized for low-latency continuous monitoring.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        segment_duration: float = 30.0,
        buffer_size: int = 10,
        ollama_model: str = "llama3.2"
    ):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.buffer_size = buffer_size

        # Processing components
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        self.classifier = EnhancedClassifier(
            n_features=39,
            ollama_model=ollama_model,
            enable_online_learning=True
        )

        # Buffers
        self.audio_buffer = np.array([])
        self.segment_samples = int(segment_duration * sample_rate)
        self.result_buffer = deque(maxlen=buffer_size)

        # State
        self.is_running = False
        self.session_id = None
        self.events_timeline = []

        # Callbacks
        self.on_event_callback = None
        self.on_alert_callback = None

    async def start_session(self, session_id: str):
        """Start a new recording session"""
        self.session_id = session_id
        self.is_running = True
        self.events_timeline = []
        self.classifier.reset_streaming()
        logger.info(f"Started session: {session_id}")

    async def stop_session(self) -> Dict:
        """Stop current session and return final results"""
        self.is_running = False

        final_stats = self.classifier.streaming_ahi.get_current_stats()
        logger.info(f"Stopped session: {self.session_id}, AHI: {final_stats['overall_ahi']}")

        return {
            'session_id': self.session_id,
            'events_timeline': self.events_timeline,
            'final_stats': final_stats
        }

    async def process_chunk(self, audio_chunk: np.ndarray) -> Optional[Dict]:
        """
        Process an incoming audio chunk.

        Args:
            audio_chunk: Raw audio samples

        Returns:
            Classification result if segment complete, None otherwise
        """
        if not self.is_running:
            return None

        # Append to buffer
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk])

        # Check if we have a complete segment
        if len(self.audio_buffer) >= self.segment_samples:
            # Extract segment
            segment = self.audio_buffer[:self.segment_samples]
            self.audio_buffer = self.audio_buffer[self.segment_samples:]

            # Process segment
            result = await self._process_segment(segment)
            return result

        return None

    async def _process_segment(self, segment: np.ndarray) -> Dict:
        """Process a complete audio segment"""
        # Preprocess
        processed = self.preprocessor.process_audio_segment(segment)

        # Extract features
        features = self.feature_extractor.extract_all_features(processed)

        # Classify
        result = self.classifier.classify(features)

        # Track event
        self.events_timeline.append(result['event_class'])

        # Store result
        self.result_buffer.append(result)

        # Check for callback
        if self.on_event_callback:
            await self.on_event_callback(result)

        # Check for alerts
        if result['ahi_stats']['overall_ahi'] >= 30 and self.on_alert_callback:
            await self.on_alert_callback(result['ahi_stats'])

        return result

    async def stream_from_websocket(
        self,
        websocket,
        chunk_duration: float = 1.0
    ) -> AsyncGenerator[Dict, None]:
        """
        Stream audio from WebSocket and yield classification results.

        Args:
            websocket: WebSocket connection
            chunk_duration: Duration of each incoming chunk

        Yields:
            Classification results
        """
        chunk_samples = int(chunk_duration * self.sample_rate)

        try:
            while self.is_running:
                # Receive audio chunk
                data = await websocket.recv()
                audio_chunk = np.frombuffer(data, dtype=np.float32)

                # Process
                result = await self.process_chunk(audio_chunk)

                if result:
                    yield result

        except Exception as e:
            logger.error(f"WebSocket streaming error: {e}")
            raise

    def get_current_stats(self) -> Dict:
        """Get current session statistics"""
        return {
            'is_running': self.is_running,
            'session_id': self.session_id,
            'segments_processed': len(self.events_timeline),
            'buffer_length': len(self.audio_buffer),
            'ahi_stats': self.classifier.streaming_ahi.get_current_stats(),
            'classifier_stats': self.classifier.get_stats()
        }

    def get_recent_results(self, n: int = 10) -> list:
        """Get N most recent classification results"""
        return list(self.result_buffer)[-n:]

    def detect_anomaly(self) -> Optional[Dict]:
        """
        Detect anomalies in recent events.
        Uses simple rule: 3+ consecutive apnea events.
        """
        if len(self.events_timeline) < 3:
            return None

        recent = self.events_timeline[-5:]
        consecutive_apnea = 0
        max_consecutive = 0

        for event in recent:
            if event == 3:  # Apnea
                consecutive_apnea += 1
                max_consecutive = max(max_consecutive, consecutive_apnea)
            else:
                consecutive_apnea = 0

        if max_consecutive >= 3:
            return {
                'type': 'consecutive_apnea',
                'count': max_consecutive,
                'message': f'Detected {max_consecutive} consecutive apnea events'
            }

        return None


class BatchProcessor:
    """
    Batch processor for full-night recordings.
    Optimized for throughput over latency.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        segment_duration: float = 30.0,
        n_workers: int = 4
    ):
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.n_workers = n_workers

        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)
        self.classifier = EnhancedClassifier(n_features=39)

    async def process_file(self, file_path: str) -> Dict:
        """
        Process a complete audio file.

        Args:
            file_path: Path to audio file

        Returns:
            Complete analysis results
        """
        # Load audio
        audio = self.preprocessor.load_audio(file_path)

        # Segment
        segments = self.preprocessor.segment_audio(audio, self.segment_duration)

        # Process all segments
        results = []
        for i, segment in enumerate(segments):
            # Preprocess
            processed = self.preprocessor.process_audio_segment(segment)

            # Extract features
            features = self.feature_extractor.extract_all_features(processed)

            # Classify
            result = self.classifier.classify(features)
            result['segment_index'] = i
            result['timestamp'] = i * self.segment_duration

            results.append(result)

            if i % 10 == 0:
                logger.info(f"Processed {i}/{len(segments)} segments")

        # Final statistics
        events_timeline = [r['event_class'] for r in results]
        final_stats = self.classifier.streaming_ahi.get_current_stats()

        return {
            'segments': results,
            'events_timeline': events_timeline,
            'final_stats': final_stats,
            'duration_hours': len(segments) * self.segment_duration / 3600
        }

    async def process_multiple(self, file_paths: list) -> list:
        """Process multiple files concurrently"""
        tasks = [self.process_file(path) for path in file_paths]
        return await asyncio.gather(*tasks)
