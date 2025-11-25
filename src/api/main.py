
import base64
import json
import logging
from datetime import datetime
from typing import List, Optional
from uuid import UUID
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.audio import AudioPreprocessor, FeatureExtractor
from src.ml import OllamaClassifier, AHICalculator
from src.utils import get_settings, AlertManager
from src.db import get_db, User, SleepSession, SleepEvent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize app
app = FastAPI(
    title="NeendAI API",
    description="Sleep Apnea Detection & Classification System",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get settings
settings = get_settings()

# Initialize components
preprocessor = AudioPreprocessor(sample_rate=settings.sample_rate)
feature_extractor = FeatureExtractor(sample_rate=settings.sample_rate)
classifier = OllamaClassifier(model=settings.ollama_model)
ahi_calculator = AHICalculator(segment_duration=settings.segment_duration)
alert_manager = AlertManager(
    alert_threshold=settings.alert_threshold,
    sns_topic_arn=settings.sns_topic_arn
)


# Pydantic models
class UserCreate(BaseModel):
    email: str
    name: Optional[str] = None
    emergency_contact: Optional[str] = None
    clinician_email: Optional[str] = None
    alert_threshold: int = 30


class SessionResponse(BaseModel):
    session_id: UUID
    start_time: datetime
    end_time: Optional[datetime]
    ahi_score: Optional[float]
    severity: Optional[str]
    total_events: int


class AnalysisResult(BaseModel):
    event_type: str
    confidence: float
    ahi: float
    severity: str
    timestamp: float


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: dict = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = {
            'websocket': websocket,
            'events': [],
            'start_time': datetime.utcnow()
        }

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]

    async def send_result(self, session_id: str, result: dict):
        if session_id in self.active_connections:
            await self.active_connections[session_id]['websocket'].send_json(result)


manager = ConnectionManager()


@app.get("/")
async def root():
    return {"status": "healthy", "service": "NeendAI API"}


@app.post("/users", response_model=dict)
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = User(
        email=user.email,
        name=user.name,
        emergency_contact=user.emergency_contact,
        clinician_email=user.clinician_email,
        alert_threshold=user.alert_threshold
    )
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return {"user_id": str(db_user.user_id), "email": db_user.email}


@app.get("/sessions/{user_id}", response_model=List[SessionResponse])
async def get_user_sessions(user_id: UUID, db: Session = Depends(get_db)):
    sessions = db.query(SleepSession).filter(
        SleepSession.user_id == user_id
    ).order_by(SleepSession.start_time.desc()).all()

    return [
        SessionResponse(
            session_id=s.session_id,
            start_time=s.start_time,
            end_time=s.end_time,
            ahi_score=s.ahi_score,
            severity=s.severity,
            total_events=s.total_events
        )
        for s in sessions
    ]


@app.websocket("/stream/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    logger.info(f"WebSocket connected: {session_id}")

    events_timeline = []

    try:
        while True:
            # Receive audio chunk
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get('type') == 'audio_chunk':
                audio_bytes = base64.b64decode(message['data'])
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                processed = preprocessor.process_audio_segment(audio_array)
                features = feature_extractor.extract_all_features(processed)
                result = classifier.classify(features)
                events_timeline.append(result['event_class'])
                ahi_result = ahi_calculator.calculate_running_ahi(events_timeline)
                response = {
                    'event_type': result['event_type'],
                    'confidence': result['confidence'],
                    'ahi': ahi_result['ahi'],
                    'severity': ahi_result['severity'],
                    'timestamp': message.get('timestamp', 0),
                    'reasoning': result.get('reasoning', ''),
                    'segments_processed': len(events_timeline)
                }
                await websocket.send_json(response)
                manager.active_connections[session_id]['events'].append(result['event_class'])
                if alert_manager.should_alert(ahi_result['ahi']):
                    logger.warning(f"Critical AHI detected: {ahi_result['ahi']}")
            elif message.get('type') == 'end_session':
                final_result = ahi_calculator.calculate_running_ahi(events_timeline)
                await websocket.send_json({
                    'type': 'session_complete',
                    'final_ahi': final_result['ahi'],
                    'severity': final_result['severity'],
                    'total_events': final_result['total_events'],
                    'apnea_events': final_result['apnea_events'],
                    'hypopnea_events': final_result['hypopnea_events']
                })
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        manager.disconnect(session_id)


@app.post("/upload-recording")
async def upload_recording(
    file: UploadFile = File(...),
    user_id: str = None,
    background_tasks: BackgroundTasks = None
):
    import tempfile
    import os

    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    # Generate job ID
    job_id = str(UUID(int=hash(f"{user_id}{datetime.utcnow().isoformat()}")))

    # Queue background processing
    background_tasks.add_task(
        process_full_recording,
        tmp_path,
        user_id,
        job_id
    )

    return {
        "status": "processing",
        "job_id": job_id,
        "message": "Recording uploaded successfully. Processing in background."
    }


async def process_full_recording(file_path: str, user_id: str, job_id: str):
    import os

    try:
        logger.info(f"Processing recording: {job_id}")

        # Load and segment audio
        audio = preprocessor.load_audio(file_path)
        segments = preprocessor.segment_audio(audio, segment_duration=settings.segment_duration)

        events_timeline = []

        for i, segment in enumerate(segments):
            # Preprocess
            processed = preprocessor.process_audio_segment(segment)

            # Extract features
            features = feature_extractor.extract_all_features(processed)

            # Classify
            result = classifier.classify(features)
            events_timeline.append(result['event_class'])

            if i % 10 == 0:
                logger.info(f"Processed segment {i}/{len(segments)}")

        # Calculate final AHI
        final_result = ahi_calculator.calculate_ahi(events_timeline)

        logger.info(f"Recording processed: AHI={final_result.ahi}, Severity={final_result.severity.value}")

        # Here you would save results to database and notify user

    except Exception as e:
        logger.error(f"Error processing recording {job_id}: {e}")
    finally:
        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)


@app.post("/analyze-segment")
async def analyze_segment(
    file: UploadFile = File(...),
):
    # Read audio file
    content = await file.read()

    # Convert to numpy array (assuming WAV format)
    import io
    import soundfile as sf

    audio_array, sample_rate = sf.read(io.BytesIO(content))

    # Resample if needed
    if sample_rate != settings.sample_rate:
        import librosa
        audio_array = librosa.resample(
            audio_array,
            orig_sr=sample_rate,
            target_sr=settings.sample_rate
        )

    # Process
    processed = preprocessor.process_audio_segment(audio_array)
    features = feature_extractor.extract_all_features(processed)
    result = classifier.classify(features)

    return {
        'event_type': result['event_type'],
        'confidence': result['confidence'],
        'reasoning': result.get('reasoning', ''),
        'features_summary': {
            'snore_score': features['snoring']['snore_score'],
            'rms_energy': features['energy']['rms_mean'],
            'spectral_centroid': features['spectral']['spectral_centroid']
        }
    }


@app.get("/reports/{session_id}")
async def get_report(session_id: UUID, db: Session = Depends(get_db)):
    session = db.query(SleepSession).filter(
        SleepSession.session_id == session_id
    ).first()

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get all events for this session
    events = db.query(SleepEvent).filter(
        SleepEvent.session_id == session_id
    ).order_by(SleepEvent.timestamp).all()

    # Create event timeline
    event_map = {'normal': 0, 'snoring': 1, 'hypopnea': 2, 'apnea': 3}
    events_timeline = [event_map.get(e.event_type, 0) for e in events]

    # Generate report
    user = db.query(User).filter(User.user_id == session.user_id).first()
    user_name = user.name if user else "Patient"

    report = ahi_calculator.generate_report(events_timeline, user_name)

    return {
        'session_id': str(session_id),
        'report': report,
        'ahi_score': session.ahi_score,
        'severity': session.severity
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
