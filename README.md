Document Anonymization using Python, Diffusion Models, and Named Entity Recognition
==================================================================================

Anonymization of sensitive information in financial documents using, python, diffusion models and named entity recognition

Data is the fossil fuel of the machine learning world, essential for developing high quality models but in limited supply. 
Yet institutions handling sensitive documents — such as financial, medical, or legal records often cannot fully leverage their own data due to stringent privacy, compliance, and security requirements, making training high quality models difficult.

A promising solution is to replace the personally identifiable information (PII) with realistic synthetic stand-ins, whilst leaving the rest of the document in tact.

In this talk, we will discuss the use of open source tools and models that can be self hosted to anonymize documents. 
We will go over the various approaches for Named Entity Recognition (NER) to identify sensitive entities and the use of diffusion models to inpaint anonymized content.


Code used for talks at Europython 2025, Pycon Lithuania 2025.

Work based off [DiffUTE](https://arxiv.org/abs/2305.10825), [REPA-E](https://github.com/End2End-Diffusion/REPA-E), [diffusers](https://github.com/huggingface/diffusers), [spaCy](https://spacy.io/), 

## Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/document-anonymizer.git
cd document-anonymizer
```

2. Install dependencies:
```bash
# Using pip
pip install -e .

# Using uv (recommended)
uv sync
```

3. Install spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

4. For cloud training, install additional dependencies:
```bash
# Modal.com and W&B for cloud training
pip install modal wandb
# or
uv add modal wandb
```

## Usage

### Anonymization (CLI)

```bash
# Single image anonymization
python main.py anonymize -c configs/inference/app_config.yaml -i input.png -o output.png

# Batch anonymization
python main.py batch-anonymize -i data/raw -o data/anonymized -c configs/inference/app_config.yaml
```

### Programmatic Usage

```python
from pathlib import Path
import numpy as np
from PIL import Image

from src.anonymizer.core.config import AppConfig
from src.anonymizer.inference.engine import InferenceEngine

app_config = AppConfig.from_env_and_yaml(yaml_path="configs/inference/app_config.yaml")
engine = InferenceEngine(app_config.engine)

image_bytes = Path("input.png").read_bytes()
result = engine.anonymize(image_bytes)

if result.success:
    Image.fromarray(result.anonymized_image.astype(np.uint8)).save("output.png")
```

### Local Training

Train models locally for development and testing:

```bash
# Train VAE locally
python main.py train-vae --config configs/training/vae_config_local.yaml

# Train UNet locally
python main.py train-unet --config configs/training/unet_config_local.yaml
```

### Cloud Training with Modal.com

For production training on high-performance GPUs:

1. **Setup Modal.com** (one-time setup):
```bash
# Install and authenticate
modal setup

# Set up secrets for W&B
modal secret create wandb-secret WANDB_API_KEY=your_wandb_api_key
```

2. **Upload training data**:
```bash
# Create volume and upload data
modal volume create anonymizer-data
modal volume mount anonymizer-data /tmp/modal-volume
cp -r /path/to/training/data /tmp/modal-volume/train/
umount /tmp/modal-volume
```

3. **Submit training jobs**:
```bash
# Train VAE on Modal.com
python scripts/train_vae_modal.py \
  --config configs/training/vae_config_cloud.yaml \
  --train-data /data/train \
  --wandb-entity your-wandb-username

# Train UNet on Modal.com
python scripts/train_unet_modal.py \
  --config configs/training/unet_config_cloud.yaml \
  --train-data /data/train \
  --wandb-entity your-wandb-username
```

4. **Monitor training**:
   - Modal Dashboard: [modal.com/apps](https://modal.com/apps)
   - W&B Dashboard: `https://wandb.ai/your-username/document-anonymizer`

For detailed setup instructions, see [docs/modal_setup.md](docs/modal_setup.md).

## Features

- **🔒 Privacy-First**: Self-hosted solution for sensitive documents
- **🚀 Cloud Training**: Scalable training on Modal.com with A100 GPUs
- **📊 Experiment Tracking**: Comprehensive logging with Weights & Biases
- **🎯 Production Ready**: Corrected hyperparameters and robust training and inference
- **🔧 Flexible**: Support for both local development and cloud deployment