# Deployment Guide

Comprehensive guide for deploying the Document Anonymization System in various environments and configurations.

## Deployment Overview

The system supports multiple deployment patterns:

1. **Standalone Desktop Application**: Single-user processing
2. **Server Deployment**: Multi-user REST API service
3. **Container Deployment**: Docker and Kubernetes
4. **Cloud Deployment**: AWS, GCP, Azure integration
5. **Serverless Deployment**: Function-as-a-Service
6. **Distributed Processing**: Multi-node processing clusters

## Prerequisites

### System Requirements

#### Minimum Requirements
- **CPU**: 4 cores, 2.5+ GHz
- **Memory**: 8GB RAM
- **Storage**: 20GB free space
- **Python**: 3.8+ (3.10+ recommended)
- **GPU** (optional): CUDA-compatible GPU with 4GB+ VRAM

#### Recommended Requirements
- **CPU**: 8+ cores, 3.0+ GHz
- **Memory**: 16GB+ RAM
- **Storage**: 100GB+ SSD
- **GPU**: NVIDIA RTX 3080 or better (8GB+ VRAM)
- **Network**: High-speed internet for model downloads

### Software Dependencies

```bash
# System packages (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    python3.10 python3.10-dev python3.10-venv \
    build-essential cmake pkg-config \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev \
    libgomp1 libfontconfig1

# CUDA support (if using GPU)
sudo apt-get install -y \
    nvidia-driver-515 \
    nvidia-cuda-toolkit \
    nvidia-cuda-dev
```

## Installation Methods

### 1. Standard Installation

```bash
# Clone repository
git clone https://github.com/your-org/document-anonymizer.git
cd document-anonymizer

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install --upgrade pip
pip install -e .

# Download required models
python scripts/download_models.py ensure-models --use-case default

# Verify installation
python -m src.anonymizer.cli --help
```

### 2. Development Installation

```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify
python -m pytest tests/ -v

# Run benchmarks
python scripts/benchmark.py full-suite --quick
```

### 3. Production Installation

```bash
# Create production user
sudo useradd -r -s /bin/false anonymizer
sudo mkdir -p /opt/anonymizer
sudo chown anonymizer:anonymizer /opt/anonymizer

# Install as service user
sudo -u anonymizer bash << 'EOF'
cd /opt/anonymizer
python3.10 -m venv venv
source venv/bin/activate
pip install --no-cache-dir document-anonymizer

# Download production models
python scripts/download_models.py ensure-models --use-case production
EOF
```

## Container Deployment

### Docker

#### Basic Dockerfile

```dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create app user
RUN useradd -r -s /bin/false anonymizer

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ src/
COPY scripts/ scripts/
COPY pyproject.toml .

# Install application
RUN pip install -e .

# Create directories
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R anonymizer:anonymizer /app

# Switch to app user
USER anonymizer

# Download models
RUN python scripts/download_models.py ensure-models --use-case default

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
    CMD python -c "from src.anonymizer.inference.engine import InferenceEngine; from src.anonymizer.core.config import AppConfig; InferenceEngine(AppConfig.from_env_and_yaml().engine); print('OK')" || exit 1

# Start application
CMD ["python", "-m", "src.anonymizer.server", "--host", "0.0.0.0", "--port", "8000"]
```

#### GPU-Enabled Dockerfile

```dockerfile
FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    build-essential cmake pkg-config \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python alias
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Continue with standard setup...
WORKDIR /app
COPY requirements-gpu.txt .
RUN pip install --no-cache-dir -r requirements-gpu.txt

# ... rest of Dockerfile
```

#### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  anonymizer:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ANONYMIZER_USE_GPU=true
      - ANONYMIZER_LOG_LEVEL=INFO
      - ANONYMIZER_MODEL_CACHE_DIR=/app/models
    volumes:
      - ./data:/app/data
      - ./models:/app/models
      - ./logs:/app/logs
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - anonymizer
    restart: unless-stopped

volumes:
  redis_data:
```

#### Building and Running

```bash
# Build image
docker build -t document-anonymizer:latest .

# Run container
docker run -d \
  --name anonymizer \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e ANONYMIZER_USE_GPU=true \
  document-anonymizer:latest

# Using Docker Compose
docker-compose up -d

# Check logs
docker-compose logs -f anonymizer
```

## Kubernetes Deployment

### Basic Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: document-anonymizer
  labels:
    app: document-anonymizer
spec:
  replicas: 3
  selector:
    matchLabels:
      app: document-anonymizer
  template:
    metadata:
      labels:
        app: document-anonymizer
    spec:
      containers:
      - name: anonymizer
        image: document-anonymizer:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: 1
        env:
        - name: ANONYMIZER_USE_GPU
          value: "true"
        - name: ANONYMIZER_LOG_LEVEL
          value: "INFO"
        - name: ANONYMIZER_MODEL_CACHE_DIR
          value: "/app/models"
        volumeMounts:
        - name: model-cache
          mountPath: /app/models
        - name: temp-storage
          mountPath: /app/temp
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 30
      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: model-cache-pvc
      - name: temp-storage
        emptyDir:
          sizeLimit: 10Gi
```

### Service Configuration

```yaml
# k8s/service.yaml
apiVersion: v1
kind: Service
metadata:
  name: document-anonymizer-service
spec:
  selector:
    app: document-anonymizer
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### ConfigMap and Secrets

```yaml
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: anonymizer-config
data:
  config.yaml: |
    anonymization:
      ocr:
        engines: ["paddleocr", "easyocr"]
        confidence_threshold: 0.8
      ner:
        entity_types: ["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"]
        confidence_threshold: 0.85
      performance:
        use_gpu: true
        batch_size: 6

---
apiVersion: v1
kind: Secret
metadata:
  name: anonymizer-secrets
type: Opaque
data:
  api_key: <base64-encoded-api-key>
  db_password: <base64-encoded-password>
```

### Persistent Storage

```yaml
# k8s/storage.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-cache-pvc
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 50Gi
```

### GPU Node Pool (GKE)

```yaml
# k8s/gpu-nodepool.yaml
apiVersion: v1
kind: Node
metadata:
  name: gpu-node-pool
spec:
  taints:
  - key: nvidia.com/gpu
    value: "true"
    effect: NoSchedule
  nodeClassRef:
    name: gpu-node-class
---
apiVersion: karpenter.k8s.aws/v1alpha1
kind: NodeClass
metadata:
  name: gpu-node-class
spec:
  instanceTypes: ["g4dn.xlarge", "g4dn.2xlarge", "p3.2xlarge"]
  amiFamily: AL2_x86_64_GPU
```

## Cloud Deployment

### AWS Deployment

#### EC2 Instance

```bash
# Launch GPU-enabled EC2 instance
aws ec2 run-instances \
  --image-id ami-0c02fb55956c7d316 \
  --instance-type g4dn.xlarge \
  --key-name my-key-pair \
  --security-group-ids sg-0123456789abcdef0 \
  --subnet-id subnet-0123456789abcdef0 \
  --user-data file://user-data.sh \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=document-anonymizer}]'
```

#### User Data Script

```bash
#!/bin/bash
# user-data.sh

# Update system
yum update -y

# Install Docker
amazon-linux-extras install docker
systemctl start docker
systemctl enable docker
usermod -a -G docker ec2-user

# Install NVIDIA drivers
yum install -y gcc kernel-devel-$(uname -r)
wget https://us.download.nvidia.com/tesla/470.57.02/NVIDIA-Linux-x86_64-470.57.02.run
bash NVIDIA-Linux-x86_64-470.57.02.run --silent

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# Deploy application
docker run -d \
  --name anonymizer \
  --gpus all \
  -p 80:8000 \
  --restart unless-stopped \
  document-anonymizer:latest
```

#### ECS Deployment

```json
{
  "family": "document-anonymizer",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["EC2"],
  "cpu": "2048",
  "memory": "8192",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::account:role/ecsTaskRole",
  "containerDefinitions": [
    {
      "name": "anonymizer",
      "image": "your-repo/document-anonymizer:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ANONYMIZER_USE_GPU",
          "value": "true"
        }
      ],
      "resourceRequirements": [
        {
          "type": "GPU",
          "value": "1"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/document-anonymizer",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Lambda Deployment

```python
# lambda_function.py
import json
import base64
from src.anonymizer.core.config import AppConfig
from src.anonymizer.inference.engine import InferenceEngine

def lambda_handler(event, context):
    """AWS Lambda handler for document anonymization."""

    try:
        # Parse request
        document_data = base64.b64decode(event['document_data'])
        config_params = event.get('config', {})

        # Initialize engine (CPU-only typical for Lambda)
        app_config = AppConfig.from_env_and_yaml()
        engine = InferenceEngine(app_config.engine)

        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_input:
            tmp_input.write(document_data)
            tmp_input_path = tmp_input.name

        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_output:
            tmp_output_path = tmp_output.name

        # Anonymize
        result = engine.anonymize(document_data)

        # Read result
        with open(tmp_output_path, 'rb') as f:
            anonymized_data = base64.b64encode(f.read()).decode('utf-8')

        # Cleanup
        os.unlink(tmp_input_path)
        os.unlink(tmp_output_path)

        return {
            'statusCode': 200,
            'body': json.dumps({
                'anonymized_document': anonymized_data,
                'processing_time_ms': result.processing_time_ms,
                'success': result.success,
                'errors': result.errors,
            })
        }

    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
```

### Google Cloud Platform

#### Cloud Run Deployment

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/PROJECT_ID/document-anonymizer

# Deploy to Cloud Run
gcloud run deploy document-anonymizer \
  --image gcr.io/PROJECT_ID/document-anonymizer \
  --platform managed \
  --region us-central1 \
  --memory 8Gi \
  --cpu 4 \
  --timeout 300 \
  --concurrency 1 \
  --max-instances 10 \
  --set-env-vars "ANONYMIZER_USE_GPU=false,ANONYMIZER_LOG_LEVEL=INFO"
```

#### GKE with GPU

```yaml
# gke-gpu-cluster.yaml
apiVersion: container.v1
kind: Cluster
metadata:
  name: anonymizer-gpu-cluster
spec:
  location: us-central1-a
  initialNodeCount: 1
  nodeConfig:
    machineType: n1-standard-4
    accelerators:
    - acceleratorCount: 1
      acceleratorType: nvidia-tesla-k80
    oauthScopes:
    - https://www.googleapis.com/auth/cloud-platform
```

### Azure Deployment

#### Container Instances

```bash
# Create resource group
az group create --name anonymizer-rg --location eastus

# Deploy container instance with GPU
az container create \
  --resource-group anonymizer-rg \
  --name document-anonymizer \
  --image myregistry.azurecr.io/document-anonymizer:latest \
  --cpu 4 \
  --memory 8 \
  --gpu-count 1 \
  --gpu-sku V100 \
  --ports 8000 \
  --dns-name-label doc-anonymizer \
  --environment-variables ANONYMIZER_USE_GPU=true
```

#### AKS Deployment

```bash
# Create AKS cluster with GPU support
az aks create \
  --resource-group anonymizer-rg \
  --name anonymizer-aks \
  --node-count 2 \
  --node-vm-size Standard_NC6s_v3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.9.0/nvidia-device-plugin.yml
```

## Service Configuration

### REST API Server

```python
# src/anonymizer/server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import tempfile
import os

app = FastAPI(title="Document Anonymization API")

@app.post("/anonymize")
async def anonymize_document(
    file: UploadFile = File(...),
):
    """Anonymize uploaded image using InferenceEngine."""

    try:
        content = await file.read()
        from src.anonymizer.core.config import AppConfig
        from src.anonymizer.inference.engine import InferenceEngine

        engine = InferenceEngine(AppConfig.from_env_and_yaml().engine)
        result = engine.anonymize(content)

        if not result.success:
            raise HTTPException(500, f"Anonymization encountered issues: {', '.join(result.errors)}")

        # For images, return the anonymized bytes as PNG
        import io
        from PIL import Image
        import numpy as np

        output = io.BytesIO()
        Image.fromarray(result.anonymized_image.astype(np.uint8)).save(output, format="PNG")
        output.seek(0)

        return Response(content=output.read(), media_type="image/png")

    except Exception as e:
        raise HTTPException(500, str(e))

    finally:
        # Cleanup
        if 'tmp_input_path' in locals():
            os.unlink(tmp_input_path)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### NGINX Configuration

```nginx
# nginx.conf
upstream anonymizer_backend {
    least_conn;
    server anonymizer1:8000;
    server anonymizer2:8000;
    server anonymizer3:8000;
}

server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name your-domain.com;

    ssl_certificate /etc/nginx/ssl/cert.pem;
    ssl_certificate_key /etc/nginx/ssl/key.pem;

    client_max_body_size 100M;
    client_body_timeout 300s;

    location / {
        proxy_pass http://anonymizer_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }

    location /health {
        proxy_pass http://anonymizer_backend/health;
        access_log off;
    }
}
```

## Monitoring and Logging

### Application Monitoring

```python
# monitoring.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time

# Metrics
REQUESTS_TOTAL = Counter('anonymization_requests_total', 'Total requests', ['status'])
REQUEST_DURATION = Histogram('anonymization_request_duration_seconds', 'Request duration')
ACTIVE_REQUESTS = Gauge('anonymization_active_requests', 'Active requests')
ENTITIES_DETECTED = Histogram('anonymization_entities_detected', 'Entities detected per document')

class MonitoringMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            start_time = time.time()
            ACTIVE_REQUESTS.inc()

            try:
                await self.app(scope, receive, send)
                REQUESTS_TOTAL.labels(status='success').inc()
            except Exception:
                REQUESTS_TOTAL.labels(status='error').inc()
                raise
            finally:
                ACTIVE_REQUESTS.dec()
                REQUEST_DURATION.observe(time.time() - start_time)
        else:
            await self.app(scope, receive, send)

# Start metrics server
start_http_server(9090)
```

### Logging Configuration

```python
# logging_config.py
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }

        if hasattr(record, 'request_id'):
            log_entry['request_id'] = record.request_id

        if hasattr(record, 'user_id'):
            log_entry['user_id'] = record.user_id

        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/app/logs/anonymizer.log')
    ]
)

# Set JSON formatter
for handler in logging.getLogger().handlers:
    handler.setFormatter(JSONFormatter())
```

### Health Checks

```python
# health.py
from typing import Dict, Any
import psutil
import torch

class HealthChecker:
    def __init__(self):
        self.checks = {
            'database': self._check_database,
            'redis': self._check_redis,
            'gpu': self._check_gpu,
            'memory': self._check_memory,
            'disk': self._check_disk,
            'models': self._check_models
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        status = {
            'status': 'healthy',
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }

        for check_name, check_func in self.checks.items():
            try:
                check_result = check_func()
                status['checks'][check_name] = {
                    'status': 'pass',
                    **check_result
                }
            except Exception as e:
                status['checks'][check_name] = {
                    'status': 'fail',
                    'error': str(e)
                }
                status['status'] = 'unhealthy'

        return status

    def _check_gpu(self) -> Dict[str, Any]:
        """Check GPU availability and memory."""
        if not torch.cuda.is_available():
            return {'available': False}

        gpu_count = torch.cuda.device_count()
        gpu_info = []

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i)
            memory_total = props.total_memory

            gpu_info.append({
                'device': i,
                'name': props.name,
                'memory_used_mb': memory_allocated // 1024**2,
                'memory_total_mb': memory_total // 1024**2,
                'memory_usage_percent': (memory_allocated / memory_total) * 100
            })

        return {
            'available': True,
            'device_count': gpu_count,
            'devices': gpu_info
        }

    def _check_memory(self) -> Dict[str, Any]:
        """Check system memory usage."""
        memory = psutil.virtual_memory()
        return {
            'total_gb': memory.total / 1024**3,
            'available_gb': memory.available / 1024**3,
            'usage_percent': memory.percent
        }
```

## Security Considerations

### SSL/TLS Configuration

```bash
# Generate self-signed certificate (development only)
openssl req -x509 -newkey rsa:4096 -nodes -keyout key.pem -out cert.pem -days 365

# Production: Use Let's Encrypt
certbot --nginx -d your-domain.com
```

### Authentication

```python
# auth.py
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token."""
    try:
        payload = jwt.decode(
            credentials.credentials,
            SECRET_KEY,
            algorithms=["HS256"]
        )
        user_id = payload.get("user_id")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
        return user_id
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# Protected endpoint
@app.post("/anonymize")
async def anonymize_document(
    file: UploadFile = File(...),
    user_id: str = Depends(verify_token)
):
    # Process document with user context
    pass
```

### Rate Limiting

```python
# rate_limiting.py
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/anonymize")
@limiter.limit("10/minute")  # 10 requests per minute
async def anonymize_document(request: Request, file: UploadFile = File(...)):
    # Process request
    pass
```

## Backup and Disaster Recovery

### Model Backup

```bash
#!/bin/bash
# backup_models.sh

BACKUP_DIR="/backup/models"
MODEL_DIR="/app/models"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p "$BACKUP_DIR/$DATE"

# Backup models with checksums
cd "$MODEL_DIR"
find . -name "*.bin" -o -name "*.safetensors" | while read file; do
    echo "Backing up $file"
    cp "$file" "$BACKUP_DIR/$DATE/"
    sha256sum "$file" >> "$BACKUP_DIR/$DATE/checksums.txt"
done

# Compress backup
tar -czf "$BACKUP_DIR/models_$DATE.tar.gz" -C "$BACKUP_DIR" "$DATE"
rm -rf "$BACKUP_DIR/$DATE"

# Upload to cloud storage
aws s3 cp "$BACKUP_DIR/models_$DATE.tar.gz" s3://your-backup-bucket/models/
```

### Application Data Backup

```bash
#!/bin/bash
# backup_data.sh

# Database backup
pg_dump anonymizer_db > /backup/db_$(date +%Y%m%d_%H%M%S).sql

# Configuration backup
tar -czf /backup/config_$(date +%Y%m%d_%H%M%S).tar.gz /app/config/

# Logs backup
tar -czf /backup/logs_$(date +%Y%m%d_%H%M%S).tar.gz /app/logs/
```

## Troubleshooting

### Common Deployment Issues

#### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi
```

#### Memory Issues

```bash
# Monitor memory usage
free -h
top -o %MEM

# Check for memory leaks
valgrind --tool=memcheck python -m src.anonymizer.cli anonymize test.pdf
```

#### Performance Issues

```bash
# Profile application
python -m cProfile -o profile.stats -m src.anonymizer.cli anonymize test.pdf
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"

# Monitor resource usage
htop
iotop
nvidia-smi -l 1
```

## Best Practices

### Deployment Checklist

- [ ] **Security**: SSL/TLS enabled, authentication configured
- [ ] **Monitoring**: Health checks, metrics, logging configured
- [ ] **Performance**: Resource limits set, GPU properly configured
- [ ] **Backup**: Regular backups scheduled, disaster recovery tested
- [ ] **Documentation**: Deployment procedures documented
- [ ] **Testing**: Smoke tests pass, performance benchmarks run
- [ ] **Scalability**: Auto-scaling configured, load balancing set up

### Production Recommendations

1. **Use dedicated GPU instances** for production workloads
2. **Implement proper load balancing** across multiple instances
3. **Set up comprehensive monitoring** and alerting
4. **Regular backup and disaster recovery testing**
5. **Implement blue-green deployments** for zero-downtime updates
6. **Use infrastructure as code** (Terraform, CloudFormation)
7. **Security scanning** of container images and dependencies
8. **Performance testing** before production deployment
