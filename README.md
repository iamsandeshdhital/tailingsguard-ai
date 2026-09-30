# TailingsGuard AI

**AI-Powered Tailings Dam Failure Prediction System**

Predicts catastrophic tailings dam failures **before they happen** — saving lives, preventing environmental disasters, and protecting mining companies from liability.

---

## The Problem

Tailings dams store toxic mining waste. When they fail, the results are catastrophic:

| Disaster | Year | Deaths | Environmental Impact |
|----------|------|--------|---------------------|
| Brumadinho, Brazil | 2019 | 270 | Doce River destroyed |
| Mount Polley, Canada | 2014 | 0 | Lake polluted |
| Ajka, Hungary | 2010 | 10 | Red sludge flood |
| Merriespruit, South Africa | 1994 | 17 | Town destroyed |

**Current monitoring is broken:**
- Manual gauges checked weekly or monthly
- No real-time early warning
- Failures happen in **minutes** — no time to react
- Engineers get blamed even when they warned management

## The Solution

TailingsGuard AI continuously fuses data from multiple sensors and predicts failures **72 hours in advance** with actionable recommendations.

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Sensor Fusion** | Piezometers, inclinometers, water level, weather, seismic |
| **ML Failure Prediction** | Gradient boosting with 72-hour prediction horizon |
| **Real-Time Dashboard** | Live health index, risk level, sensor status |
| **Smart Alerts** | Email, SMS, siren with escalation |
| **Compliance Reporting** | Automated regulatory reports |
| **Data Logging** | SQLite with 1-year retention |
| **Simulation Mode** | Test failure scenarios safely |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run in Demo Mode
```bash
python tailingsguard.py --demo
```

### 3. Run Tests
```bash
python -m pytest tests/ -v --cov=core
```

### 4. Docker Deployment
```bash
docker-compose up -d
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TailingsGuard AI                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ Piezometers │  │Inclinometers│  │ Water Level │         │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘         │
│         │                │                │                 │
│         └────────────────┼────────────────┘                 │
│                          ▼                                  │
│              ┌─────────────────────┐                       │
│              │   Sensor Fusion     │                       │
│              │   Health Index      │                       │
│              └──────────┬──────────┘                       │
│                         ▼                                   │
│              ┌─────────────────────┐                       │
│              │  Failure Predictor  │                       │
│              │  (ML Model)         │                       │
│              └──────────┬──────────┘                       │
│                         ▼                                   │
│         ┌───────────────┼───────────────┐                  │
│         ▼               ▼               ▼                  │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐          │
│  │  Dashboard │  │   Alerts   │  │Data Logger │          │
│  └────────────┘  └────────────┘  └────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Project Structure

```
tailingsguard-ai/
├── tailingsguard.py          # Main application
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── Dockerfile                # Container definition
├── docker-compose.yml        # Multi-service deployment
├── Makefile                  # Build automation
├── README.md                 # This file
├── core/
│   ├── __init__.py
│   ├── sensor_fusion.py      # Multi-sensor data fusion
│   ├── failure_predictor.py  # ML failure prediction
│   ├── alert_system.py       # Alert management
│   ├── dashboard.py          # Real-time dashboard
│   ├── data_logger.py        # Data persistence
│   └── simulator.py          # Sensor simulation
├── tests/
│   ├── test_sensor_fusion.py
│   ├── test_failure_predictor.py
│   ├── test_alert_system.py
│   ├── test_data_logger.py
│   └── test_simulator.py
├── data/                     # Database files
├── models/                   # Trained ML models
└── .github/
    └── workflows/
        └── ci.yml            # CI/CD pipeline
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Current dam health index |
| `/api/sensors` | GET | All sensor readings |
| `/api/prediction` | GET | Current failure prediction |
| `/api/alerts` | GET | Active alerts |
| `/api/trends` | GET | Historical trends |
| `/api/report` | POST | Generate compliance report |

## Configuration

Edit `config.yaml` to customize:

```yaml
dam:
  height_m: 45
  type: upstream

safety:
  fs_minimum: 1.5
  fs_warning: 1.3
  fs_critical: 1.1

alerts:
  recipients:
    - name: "Mine Manager"
      email: "manager@mine.com"
      phone: "+1234567890"
```

## Safety Thresholds

| Parameter | Safe | Warning | Critical |
|-----------|------|---------|----------|
| Factor of Safety | > 1.5 | 1.3 - 1.5 | < 1.3 |
| Phreatic Ratio | < 50% | 50 - 70% | > 70% |
| Movement Rate | < 5 mm/day | 5 - 20 mm/day | > 20 mm/day |
| Beach Width | > 100m | 50 - 100m | < 50m |

## Testing

```bash
# Run all tests
make test

# Run with coverage
python -m pytest tests/ -v --cov=core --cov-report=html

# Run specific test
python -m pytest tests/test_sensor_fusion.py -v
```

## Docker Deployment

```bash
# Build image
docker build -t tailingsguard-ai:latest .

# Run container
docker run -p 8050:8050 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  tailingsguard-ai:latest

# Or use docker-compose
docker-compose up -d
```

## CI/CD

The project includes GitHub Actions workflows for:
- Automated testing on Python 3.9, 3.10, 3.11
- Code linting with flake8
- Security scanning with bandit
- Docker image building
- Code coverage reporting

## License

MIT License — See LICENSE file for details.

## Disclaimer

This system is designed to **assist** qualified engineers, not replace them. Always follow local regulations and consult with geotechnical engineers for dam safety decisions.
