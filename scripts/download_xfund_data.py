#!/usr/bin/env python3
"""
Download and prepare XFUND dataset for VAE and UNET training
============================================================

This script downloads the XFUND dataset from GitHub releases and prepares it
for training document anonymization models (VAE and UNET).

XFUND is a multilingual dataset for form understanding with various document types.
"""

import argparse
import json
import logging
import tarfile
import zipfile
from pathlib import Path
from typing import ClassVar
from urllib.request import urlretrieve

import cv2
import numpy as np
from tqdm import tqdm

# Constants
BOX_COORDINATES_COUNT = 4

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class XFUNDDownloader:
    """Downloads XFUND dataset from GitHub releases."""

    BASE_URL = "https://github.com/doc-analysis/XFUND/releases/download/v1.0"
    LANGUAGES: ClassVar[list[str]] = ["zh", "ja", "es", "fr", "it", "de", "pt"]

    def __init__(self, download_dir: Path, languages: list[str] | None = None):
        self.download_dir = download_dir
        self.languages = languages or ["zh"]  # Default to Chinese dataset
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.downloaded_languages = []

    def download_dataset(self) -> dict[str, Path]:
        """Download selected language datasets."""
        downloaded_files = {}

        for lang in self.languages:
            logger.info(f"Downloading XFUND dataset for language: {lang}")

            # Download JSON files
            for split in ["train", "val"]:
                filename = f"{lang}.{split}.json"
                url = f"{self.BASE_URL}/{filename}"
                output_path = self.download_dir / filename

                if output_path.exists():
                    logger.info(f"File already exists: {output_path}")
                else:
                    try:
                        logger.info(f"Downloading from: {url}")
                        urlretrieve(
                            url, output_path, reporthook=self._download_progress
                        )  # noqa: S310
                        logger.info(f"Downloaded: {output_path}")
                    except Exception:
                        logger.exception(f"Failed to download {filename}")
                        continue

            # Download images (zip file)
            img_filename = f"{lang}.train.zip"
            img_url = f"{self.BASE_URL}/{img_filename}"
            img_output_path = self.download_dir / img_filename

            if img_output_path.exists():
                logger.info(f"Image archive already exists: {img_output_path}")
            else:
                try:
                    logger.info(f"Downloading images from: {img_url}")
                    urlretrieve(
                        img_url, img_output_path, reporthook=self._download_progress
                    )  # noqa: S310
                    logger.info(f"Downloaded: {img_output_path}")
                except Exception:
                    logger.exception(f"Failed to download images for {lang}")
                    continue

            # Extract images
            if img_output_path.exists():
                self._extract_images(img_output_path, lang, "zip")
                downloaded_files[lang] = self.download_dir / f"{lang}.train.json"
                self.downloaded_languages.append(lang)

        return downloaded_files

    def _extract_images(self, archive_path: Path, lang: str, format_type: str):
        """Extract images from archive."""
        extract_dir = self.download_dir / f"{lang}_images"
        extract_dir.mkdir(exist_ok=True)

        try:
            if format_type == "zip":
                with zipfile.ZipFile(archive_path, "r") as zip_ref:
                    zip_ref.extractall(extract_dir)  # noqa: S202
            elif format_type == "tar.gz":
                with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(extract_dir)  # noqa: S202
            logger.info(f"Extracted images to: {extract_dir}")
        except Exception:
            logger.exception(f"Failed to extract {archive_path}")

    def _download_progress(self, block_num: int, block_size: int, total_size: int):
        """Show download progress."""
        downloaded = block_num * block_size
        percent = min(100, (downloaded / total_size) * 100)
        logger.info(f"Download progress: {percent:.1f}%")


class XFUNDProcessor:
    """Process XFUND data for VAE and UNET training."""

    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        self.vae_dir = self.output_dir / "vae"
        self.unet_dir = self.output_dir / "unet"

        # Create train/val subdirectories
        for base_dir in [self.vae_dir, self.unet_dir]:
            (base_dir / "train").mkdir(parents=True, exist_ok=True)
            (base_dir / "val").mkdir(parents=True, exist_ok=True)

        self.processed_languages = []

    def process_dataset(self, languages: list[str]):
        """Process downloaded datasets for all languages."""
        for lang in languages:
            logger.info(f"Processing {lang} dataset")

            # Load annotations
            json_path = self.data_dir / f"{lang}.train.json"
            if not json_path.exists():
                logger.error(f"Annotation file not found: {json_path}")
                continue

            with json_path.open(encoding="utf-8") as f:
                data = json.load(f)

            # Process documents
            documents = data.get("documents", [])
            logger.info(f"Processing {len(documents)} documents")

            for doc in tqdm(documents, desc=f"Processing {lang}"):
                self._process_document(doc, lang)

            self.processed_languages.append(lang)

    def _process_document(self, doc: dict, lang: str):
        """Process a single document for training."""
        try:
            # Get image path
            img_info = doc.get("img", {})
            img_fname = img_info.get("fname")
            if not img_fname:
                logger.warning("No image filename found")
                return

            img_path = self.data_dir / f"{lang}_images" / img_fname

            # Check for image existence
            if not img_path.exists():
                # Try without subdirectory
                img_path = self.data_dir / img_fname
                if not img_path.exists():
                    logger.debug(f"Image not found: {img_fname}, creating synthetic data")
                    # Create synthetic image based on document size
                    width = img_info.get("width", 1000)
                    height = img_info.get("height", 1000)
                    image = self._create_synthetic_image(width, height, doc)
                else:
                    image = cv2.imread(str(img_path))
            else:
                # Load image
                image = cv2.imread(str(img_path))

            if image is None:
                logger.warning(f"Failed to load image: {img_path}")
                return

            # Process for VAE training
            self._prepare_vae_data(image, doc, lang)

            # Process for UNET training
            self._prepare_unet_data(image, doc, lang)

        except Exception:
            logger.exception("Failed to process document")

    def _create_synthetic_image(self, width: int, height: int, doc: dict) -> np.ndarray:
        """Create a synthetic document image for testing when real images are not available."""
        # Create white background
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw bounding boxes for text regions
        for item in doc.get("document", []):
            box = item.get("box", [])
            if len(box) == BOX_COORDINATES_COUNT:
                x1, y1, x2, y2 = box
                # Draw light gray box to represent text region
                cv2.rectangle(image, (x1, y1), (x2, y2), (200, 200, 200), -1)
                # Draw black border
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), 1)

        return image

    def _prepare_vae_data(self, image: np.ndarray, doc: dict, lang: str):
        """Prepare data for VAE training."""
        # VAE typically needs:
        # 1. Original images
        # 2. Images with text regions masked/removed

        doc_id = doc.get("id", "unknown")

        # Save original image
        original_path = self.vae_dir / f"{lang}_{doc_id}_original.png"
        cv2.imwrite(str(original_path), image)

        # Create masked version
        masked_image = image.copy()

        # Get text regions from document
        for item in doc.get("document", []):
            box = item.get("box", [])
            if len(box) == BOX_COORDINATES_COUNT:
                x1, y1, x2, y2 = box
                # Mask text region with white
                cv2.rectangle(masked_image, (x1, y1), (x2, y2), (255, 255, 255), -1)

        # Save masked image
        masked_path = self.vae_dir / f"{lang}_{doc_id}_masked.png"
        cv2.imwrite(str(masked_path), masked_image)

        # Save metadata (compatible with AnonymizerDataset)
        metadata = {
            "image_name": str(original_path.name),  # Primary image for dataset loader
            "original": str(original_path.name),
            "masked": str(masked_path.name),
            "language": lang,
            "doc_id": doc_id,
            "text_regions": [],
        }

        # Extract text regions
        for item in doc.get("document", []):
            box = item.get("box", [])
            text = item.get("text", "")
            label = item.get("label", "")

            if len(box) == BOX_COORDINATES_COUNT:
                metadata["text_regions"].append(
                    {
                        "bbox": {
                            "left": box[0],
                            "top": box[1],
                            "right": box[2],
                            "bottom": box[3],
                        },
                        "original_text": text,
                        "label": label,
                        "replacement_text": f"[{label}]" if label else "[REDACTED]",
                        "confidence": 1.0,  # Add confidence score
                    }
                )

        # Save metadata
        metadata_path = self.vae_dir / f"{lang}_{doc_id}_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _prepare_unet_data(self, image: np.ndarray, doc: dict, lang: str):
        """Prepare data for UNET training."""
        # UNET typically needs:
        # 1. Input images
        # 2. Segmentation masks for text regions

        doc_id = doc.get("id", "unknown")
        height, width = image.shape[:2]

        # Save input image
        input_path = self.unet_dir / f"{lang}_{doc_id}_input.png"
        cv2.imwrite(str(input_path), image)

        # Create segmentation mask
        mask = np.zeros((height, width), dtype=np.uint8)

        # Different classes for different entity types
        label_to_class = {
            "PERSON": 1,
            "ORGANIZATION": 2,
            "LOCATION": 3,
            "DATE": 4,
            "MONEY": 5,
            "OTHER": 6,
        }

        # Create mask with different classes
        for item in doc.get("document", []):
            box = item.get("box", [])
            label = item.get("label", "OTHER")

            if len(box) == BOX_COORDINATES_COUNT:
                x1, y1, x2, y2 = box
                class_id = label_to_class.get(label, 6)
                cv2.rectangle(mask, (x1, y1), (x2, y2), class_id, -1)

        # Save mask
        mask_path = self.unet_dir / f"{lang}_{doc_id}_mask.png"
        cv2.imwrite(str(mask_path), mask)

        # Save metadata
        metadata = {
            "input": str(input_path.name),
            "mask": str(mask_path.name),
            "language": lang,
            "doc_id": doc_id,
            "classes": label_to_class,
            "image_size": [width, height],
        }

        metadata_path = self.unet_dir / f"{lang}_{doc_id}_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def create_split_files(self, train_ratio: float = 0.8):
        """Create train/val split files."""
        for data_type in ["vae", "unet"]:
            data_dir = self.vae_dir if data_type == "vae" else self.unet_dir

            # Get all metadata files
            metadata_files = list(data_dir.glob("*_metadata.json"))

            # Shuffle and split
            np.random.shuffle(metadata_files)
            split_idx = int(len(metadata_files) * train_ratio)

            train_files = metadata_files[:split_idx]
            val_files = metadata_files[split_idx:]

            # Save split files
            train_list = [str(f.name) for f in train_files]
            val_list = [str(f.name) for f in val_files]

            with (data_dir / "train_files.json").open("w") as f:
                json.dump(train_list, f, indent=2)

            with (data_dir / "val_files.json").open("w") as f:
                json.dump(val_list, f, indent=2)

            logger.info(f"{data_type.upper()} split: {len(train_list)} train, {len(val_list)} val")


def main():
    """Main function to download and prepare XFUND data."""
    parser = argparse.ArgumentParser(description="Download and prepare XFUND dataset")
    parser.add_argument(
        "--languages",
        nargs="+",
        default=["zh"],
        choices=["zh", "ja", "es", "fr", "it", "de", "pt"],
        help="Languages to download (zh=Chinese, ja=Japanese, es=Spanish, fr=French, it=Italian, de=German, pt=Portuguese)",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/raw/xfund"),
        help="Directory to download raw data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/xfund"),
        help="Directory for processed data",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of training data")

    args = parser.parse_args()

    # Download data
    logger.info("Starting XFUND dataset download")
    downloader = XFUNDDownloader(args.download_dir, args.languages)
    downloaded = downloader.download_dataset()

    if not downloaded:
        logger.error("No datasets downloaded")
        return

    # Process data
    logger.info("Processing downloaded data")
    processor = XFUNDProcessor(args.download_dir, args.output_dir)
    processor.process_dataset(args.languages)

    # Create train/val splits
    logger.info("Creating train/validation splits")
    processor.create_split_files(args.train_ratio)

    logger.info("Data preparation complete!")
    logger.info(f"VAE data: {processor.vae_dir}")
    logger.info(f"UNET data: {processor.unet_dir}")


if __name__ == "__main__":
    main()
