# SolarCast Deployment Guide

## Local Development Setup

To run locally using docker-compose:
```bash
cd deployment
docker-compose up --build
```
This will start the backend at `http://localhost:8000` and frontend at `http://localhost:3000`.

## Docker Deployment

To build individual containers:

### Backend
```bash
docker build -t solarcast-backend -f deployment/Dockerfile .
docker run -p 8000:8000 solarcast-backend
```

### Frontend
```bash
cd frontend
docker build -t solarcast-frontend -f ../deployment/Dockerfile.frontend .
docker run -p 3000:3000 solarcast-frontend
```

## Vercel Frontend Deployment

1. Install Vercel CLI: `npm i -g vercel`
2. Run `vercel` in the `frontend` directory.
3. Configure `NEXT_PUBLIC_API_URL` in the Vercel dashboard.

## Backend Deployment Options

You can deploy the backend Docker container to any standard PAAS:
- **Railway**: Connect repo, specify `deployment/Dockerfile`
- **Render**: Create Web Service, select Docker, specify `deployment/Dockerfile`
- **Fly.io**: `fly launch` and point to the Dockerfile.

### Environment Variables
- `MODEL_PATH`: Path to model weights
- `USE_ONNX`: `true` or `false`
- `CORS_ORIGINS`: Comma-separated allowed origins for frontend (e.g., `https://solarcast.vercel.app`)
