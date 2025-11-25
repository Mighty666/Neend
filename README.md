# NeendAI - Sleep Apnea Detection System

sleep apnea detection using audio analysis. started in sept 2023.

main result: auroc of 0.942 (95% ci: 0.935-0.949) on test set

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Next.js](https://img.shields.io/badge/Next.js-14-black.svg)

## what this is

detects sleep apnea from audio recordings. uses:
- transformer model trained on sleep recordings
- audio features from literature
- statistical testing
- distributed preprocessing

web and mobile apps for users, research/ folder has ml code.

## key results

| metric | value | 95% ci |
|--------|-------|--------|
| auroc | 0.942 | 0.935-0.949 |
| sensitivity | 0.891 | 0.878-0.904 |
| specificity | 0.923 | 0.912-0.934 |
| ppv | 0.887 | 0.871-0.903 |
| npv | 0.926 | 0.915-0.937 |

tested on 571 held-out recordings (20% of total dataset)

## datasets

combined 5 public sleep datasets. took forever to get the licensing sorted out.

| dataset | recordings | hours | source |
|---------|------------|-------|--------|
| sleep-edf | 197 | ~1,500 | physionet |
| shhs | 1,423 | ~11,000 | nsrr |
| physionet apnea-ecg | 70 | ~560 | physionet |
| a3 dataset | 432 | ~3,400 | github |
| cosmos | 725 | ~5,800 | request |
| **total** | **2,847** | **~22,260** | |

data splits: 60% train (1,708), 20% val (568), 20% test (571)

## model

transformer model:
- 24 layers, 1024 hidden dim
- pretrained with masked spectrogram modeling
- 400k steps on 2,847 recordings

## audio features

mfccs, mel spectrograms, spectral features, energy features, formants, pitch

## pretraining

tried wav2vec, hubert, byol-a. ended up with masked spectrogram modeling.

## hyperparameter search

ran 1,247 optuna trials. best config: lr 3.2e-4, batch 64, 24 layers

## statistical analysis

bootstrap confidence intervals, delong test, mcnemar test, cohen's d

## features

web app: dark theme, audio recording, ahi visualization, history
mobile app: react native, audio recording, offline support
backend: fastapi, websockets, postgresql

## Quick Start

### Using Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/aranyoray/neendAI.git
cd neendAI

# Start all services
docker-compose up -d

# Access the apps
# API: http://localhost:8000
# Web: http://localhost:3000
```

### Manual Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install and start Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Start the API server
uvicorn src.api.main:app --reload --port 8000

# In another terminal, start the web app
cd web && npm install && npm run dev

# For mobile app
cd mobile && npm install && npx expo start
```

## Project Structure

```
neendAI/
├── src/                          # Python backend
│   ├── audio/                    # Audio processing
│   │   ├── preprocessor.py       # Noise reduction, filtering
│   │   └── features.py           # MFCC, FFT, spectral features
│   ├── ml/                       # Machine learning
│   │   ├── classifier.py         # Ollama-based classification
│   │   ├── ahi_calculator.py     # AHI scoring
│   │   ├── enhanced_classifier.py # Ensemble with online learning
│   │   ├── online/               # Streaming algorithms
│   │   │   └── algorithms.py     # SGD, Welford, AdaGrad
│   │   ├── advanced_analytics.py # Trends, correlations, risk
│   │   ├── real_time_processor.py # Streaming pipeline
│   │   ├── interpretability.py   # Explainable predictions
│   │   └── model_training.py     # Training pipelines
│   ├── api/                      # FastAPI backend
│   │   ├── main.py               # WebSocket & REST endpoints
│   │   └── auth.py               # Authentication
│   ├── db/                       # Database
│   │   ├── models.py             # SQLAlchemy models
│   │   └── database.py           # Connection management
│   ├── data/                     # Dataset utilities
│   │   └── datasets.py           # PhysioNet, augmentation
│   └── utils/                    # Utilities
│       ├── config.py             # Settings
│       └── alerts.py             # Notification system
├── web/                          # Next.js web app
│   ├── app/                      # App router pages
│   │   ├── page.tsx              # Landing page
│   │   ├── auth/                 # Sign in/up flows
│   │   ├── dashboard/            # Main dashboard
│   │   ├── recording/            # Recording view
│   │   ├── history/              # Session history
│   │   ├── reports/              # Detailed reports
│   │   └── settings/             # User settings
│   ├── components/               # React components
│   │   ├── ui/                   # Reusable UI
│   │   ├── AHIGauge.tsx          # Circular gauge
│   │   ├── BreathingVisual.tsx   # Animation
│   │   ├── EventTimeline.tsx     # Event bars
│   │   └── SleepQualityScore.tsx # Quality ring
│   └── lib/                      # Utilities
│       ├── store.ts              # Zustand state
│       └── utils.ts              # Helpers
├── mobile/                       # React Native app
│   ├── app/                      # Expo Router screens
│   │   ├── index.tsx             # Welcome
│   │   ├── auth/                 # Authentication
│   │   ├── dashboard.tsx         # Dashboard
│   │   ├── recording.tsx         # Recording
│   │   ├── history.tsx           # History
│   │   └── settings.tsx          # Settings
│   ├── components/               # Native components
│   │   ├── BottomNav.tsx         # Navigation
│   │   ├── AHIRing.tsx           # Gauge
│   │   └── EventBars.tsx         # Visualization
│   └── lib/                      # Utilities
│       ├── store.ts              # State
│       ├── api.ts                # API client
│       └── websocket.ts          # Streaming
├── tests/                        # Unit tests
├── examples/                     # Example scripts
├── docker-compose.yml            # Docker services
├── Dockerfile                    # API container
└── requirements.txt              # Python dependencies
```

## API Endpoints

### REST
- `POST /users` - Create user account
- `POST /auth/signup` - Sign up with email validation
- `POST /auth/signin` - Sign in with credentials
- `GET /sessions/{user_id}` - Get user's sleep sessions
- `GET /reports/{session_id}` - Generate clinical report
- `POST /upload-recording` - Upload full recording
- `POST /analyze-segment` - Analyze single segment

### WebSocket
- `WS /stream/{session_id}` - Real-time audio streaming

## Classification

| Class | Type | Description |
|-------|------|-------------|
| 0 | Normal | Regular breathing pattern |
| 1 | Snoring | Periodic vibrations, 20-300 Hz |
| 2 | Hypopnea | Partial obstruction, reduced airflow |
| 3 | Apnea | Complete cessation >10 seconds |

## AHI Severity

| AHI Range | Severity | Action |
|-----------|----------|--------|
| < 5 | Normal | Continue monitoring |
| 5-15 | Mild | Lifestyle changes |
| 15-30 | Moderate | Consult specialist |
| ≥ 30 | Severe | Urgent evaluation |

## tech stack

research: pytorch, ray, dask, optuna, librosa
backend: python, fastapi, postgresql, ollama
web: next.js, typescript, tailwind
mobile: expo, react native


## Development

```bash
# Run tests
pytest tests/ -v

# Format code
black src/
isort src/

# Type check
mypy src/

# Web development
cd web && npm run dev

# Mobile development
cd mobile && npx expo start
```

## Configuration

Environment variables (`.env`):

```env
# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/neendai

# Ollama
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.2

# Alerts
ALERT_THRESHOLD=30
SNS_TOPIC_ARN=arn:aws:sns:...

# Security
SECRET_KEY=your-secret-key
```

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open Pull Request

## License

MIT License - see [LICENSE](LICENSE) for details.

## compute

pretraining: ~2000 gpu hours
hyperparameter search: ~800 gpu hours
total cost: ~$6000-7000

## references

see research/literature/ for bibliography


## license

mit license - see [LICENSE](LICENSE) for details.

