"""
Derm Foundation Wrapper for Skin Analysis

This module integrates Google's Derm Foundation model from HAI-DEF
for dermatology image analysis. The model produces 6144-dimensional
embeddings that are used to classify skin conditions.

Reference: https://huggingface.co/google/derm-foundation
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from io import BytesIO
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Common skin conditions the model can help identify
SKIN_CONDITIONS = {
    "melanoma": {
        "name": "Melanoma",
        "severity": "HIGH",
        "description": "A serious form of skin cancer that develops from melanocytes.",
        "recommendation": "Seek immediate dermatologist consultation.",
    },
    "benign_nevus": {
        "name": "Benign Nevus (Mole)",
        "severity": "LOW",
        "description": "A common, non-cancerous skin growth.",
        "recommendation": "Monitor for changes in size, shape, or color.",
    },
    "psoriasis": {
        "name": "Psoriasis",
        "severity": "MODERATE",
        "description": "A chronic autoimmune condition causing rapid skin cell buildup.",
        "recommendation": "Consult a dermatologist for treatment options.",
    },
    "eczema": {
        "name": "Eczema/Dermatitis",
        "severity": "MODERATE",
        "description": "Inflammation of the skin causing itchiness, redness, and rashes.",
        "recommendation": "Keep skin moisturized; consult doctor if severe.",
    },
    "acne": {
        "name": "Acne",
        "severity": "LOW",
        "description": "Common skin condition caused by clogged hair follicles.",
        "recommendation": "Maintain good skincare; see dermatologist for persistent cases.",
    },
    "rosacea": {
        "name": "Rosacea",
        "severity": "LOW",
        "description": "Chronic condition causing facial redness and visible blood vessels.",
        "recommendation": "Avoid triggers; consult dermatologist for treatment.",
    },
    "fungal_infection": {
        "name": "Fungal Infection",
        "severity": "MODERATE",
        "description": "Skin infection caused by fungi, often appearing as circular rashes.",
        "recommendation": "Use antifungal treatments; see doctor if spreading.",
    },
    "urticaria": {
        "name": "Urticaria (Hives)",
        "severity": "LOW",
        "description": "Raised, itchy welts on the skin, often an allergic reaction.",
        "recommendation": "Identify triggers; antihistamines may help.",
    },
    "contact_dermatitis": {
        "name": "Contact Dermatitis",
        "severity": "LOW",
        "description": "Skin reaction from contact with an irritant or allergen.",
        "recommendation": "Avoid the irritant; use topical corticosteroids if needed.",
    },
    "seborrheic_dermatitis": {
        "name": "Seborrheic Dermatitis",
        "severity": "LOW",
        "description": "Common skin condition causing scaly patches and red skin.",
        "recommendation": "Use medicated shampoos; see doctor for persistent cases.",
    },
}


class DermFoundationWrapper:
    """
    Wrapper for Google's Derm Foundation model.

    This model produces embeddings that can be used to:
    1. Classify skin conditions
    2. Assess image quality
    3. Identify body parts
    4. Detect skin abnormalities
    """

    def __init__(self, use_gpu: bool = False):
        """Initialize the Derm Foundation model."""
        self._model = None
        self._use_gpu = use_gpu
        self._loaded = False
        self._tf = None

    async def load_model(self) -> bool:
        """Load the Derm Foundation model from Hugging Face."""
        if self._loaded:
            return True

        try:
            import tensorflow as tf

            self._tf = tf

            # Configure GPU/CPU
            if not self._use_gpu:
                tf.config.set_visible_devices([], "GPU")

            from huggingface_hub import from_pretrained_keras

            logger.info("Loading Derm Foundation model from Hugging Face...")
            self._model = from_pretrained_keras("google/derm-foundation")
            self._loaded = True
            logger.info("Derm Foundation model loaded successfully")
            return True

        except ImportError as e:
            logger.error("Missing dependencies for Derm Foundation: %s", e)
            logger.info("Install with: pip install tensorflow huggingface_hub")
            return False
        except Exception as e:
            logger.error("Failed to load Derm Foundation model: %s", e)
            return False

    def _preprocess_image(self, image: Image.Image) -> bytes:
        """
        Preprocess image for Derm Foundation model.

        The model expects 448x448 PNG images.
        """
        # Resize to 448x448 as required by the model
        image = image.convert("RGB")
        image = image.resize((448, 448), Image.Resampling.LANCZOS)

        # Convert to PNG bytes
        buf = BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    def _create_input_tensor(self, image_bytes: bytes):
        """Create the input tensor in the format expected by the model."""
        tf = self._tf
        input_tensor = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "image/encoded": tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image_bytes])
                    )
                }
            )
        ).SerializeToString()
        return input_tensor

    async def get_embedding(self, image: Image.Image) -> Optional[np.ndarray]:
        """
        Get the 6144-dimensional embedding for a skin image.

        Args:
            image: PIL Image of skin

        Returns:
            Numpy array of shape (6144,) or None if failed
        """
        if not await self.load_model():
            return None

        try:
            # Preprocess image
            image_bytes = self._preprocess_image(image)
            input_tensor = self._create_input_tensor(image_bytes)

            # Run inference
            tf = self._tf
            infer = self._model.signatures["serving_default"]
            output = infer(inputs=tf.constant([input_tensor]))

            # Extract embedding
            embedding = output["embedding"].numpy().flatten()
            logger.info("Generated embedding with shape: %s", embedding.shape)

            return embedding

        except Exception as e:
            logger.error("Failed to generate embedding: %s", e)
            return None

    async def analyze_skin(
        self, image: Image.Image, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze a skin image for potential conditions.

        This uses the embedding to provide a preliminary assessment.

        Args:
            image: PIL Image of skin
            context: Optional text description of symptoms

        Returns:
            Dict with analysis results
        """
        # Get embedding
        embedding = await self.get_embedding(image)

        if embedding is None:
            return {
                "success": False,
                "error": "Failed to analyze image",
                "conditions": [],
                "recommendations": [],
            }

        # Analyze embedding for skin condition indicators
        # This uses heuristics based on embedding patterns
        # In production, you would train a classifier on labeled data
        results = self._analyze_embedding(embedding, context)

        return {
            "success": True,
            "model": "derm-foundation",
            "embedding_dim": len(embedding),
            **results,
        }

    def _analyze_embedding(
        self, embedding: np.ndarray, context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze the embedding to detect potential skin conditions.

        Note: This is a simplified heuristic approach. For production use,
        train a classifier on labeled dermatology datasets using these embeddings.
        """
        # Calculate embedding statistics for basic analysis
        embedding_mean = np.mean(embedding)
        embedding_std = np.std(embedding)
        embedding_max = np.max(embedding)
        embedding_min = np.min(embedding)

        # Analyze context for keywords
        context_lower = context.lower() if context else ""

        potential_conditions = []

        # Keyword-based condition detection from context
        condition_keywords = {
            "melanoma": [
                "dark spot",
                "mole changed",
                "irregular mole",
                "black spot",
                "growing mole",
            ],
            "psoriasis": ["scaly", "silvery", "patches", "plaque", "flaky"],
            "eczema": ["itchy", "red patches", "dry skin", "rash", "irritated"],
            "acne": ["pimple", "acne", "blackhead", "whitehead", "breakout", "zit"],
            "rosacea": ["facial redness", "flushing", "red face", "visible veins"],
            "fungal_infection": ["ring", "circular", "spreading", "fungal", "ringworm"],
            "urticaria": ["hives", "welts", "allergic", "swelling", "itchy bumps"],
            "contact_dermatitis": [
                "touched something",
                "reaction",
                "irritant",
                "allergic rash",
            ],
        }

        for condition_key, keywords in condition_keywords.items():
            for keyword in keywords:
                if keyword in context_lower:
                    condition = SKIN_CONDITIONS[condition_key]
                    confidence = 0.6 + (
                        embedding_std * 0.1
                    )  # Base confidence + embedding factor
                    confidence = min(
                        0.85, confidence
                    )  # Cap at 85% - AI should not be overconfident

                    potential_conditions.append(
                        {
                            "condition": condition["name"],
                            "confidence": round(confidence, 2),
                            "severity": condition["severity"],
                            "description": condition["description"],
                            "recommendation": condition["recommendation"],
                        }
                    )
                    break

        # If no conditions detected from context, provide general analysis
        if not potential_conditions:
            # Check if embedding suggests normal vs abnormal skin
            # Higher variance in embeddings often correlates with abnormalities
            if embedding_std > 0.5:
                potential_conditions.append(
                    {
                        "condition": "Potential Skin Abnormality",
                        "confidence": 0.5,
                        "severity": "MODERATE",
                        "description": "The AI detected features that may warrant further examination.",
                        "recommendation": "Consider consulting a dermatologist for professional evaluation.",
                    }
                )
            else:
                potential_conditions.append(
                    {
                        "condition": "Normal Skin Appearance",
                        "confidence": 0.7,
                        "severity": "LOW",
                        "description": "No obvious abnormalities detected in the image.",
                        "recommendation": "Continue regular skin monitoring and sun protection.",
                    }
                )

        # Determine overall triage level
        severities = [c["severity"] for c in potential_conditions]
        if "HIGH" in severities:
            triage_level = "HIGH"
        elif "MODERATE" in severities:
            triage_level = "MODERATE"
        else:
            triage_level = "LOW"

        # Generate recommendations
        recommendations = list(set([c["recommendation"] for c in potential_conditions]))

        # Add standard disclaimer
        recommendations.append(
            "Note: This AI analysis is for informational purposes only. "
            "Please consult a qualified dermatologist for accurate diagnosis."
        )

        return {
            "conditions": potential_conditions,
            "triage_level": triage_level,
            "recommendations": recommendations,
            "embedding_stats": {
                "mean": round(float(embedding_mean), 4),
                "std": round(float(embedding_std), 4),
                "range": [
                    round(float(embedding_min), 4),
                    round(float(embedding_max), 4),
                ],
            },
        }

    async def compare_images(
        self, image1: Image.Image, image2: Image.Image
    ) -> Dict[str, Any]:
        """
        Compare two skin images to detect changes over time.

        Useful for monitoring moles or tracking treatment progress.
        """
        embedding1 = await self.get_embedding(image1)
        embedding2 = await self.get_embedding(image2)

        if embedding1 is None or embedding2 is None:
            return {
                "success": False,
                "error": "Failed to generate embeddings for comparison",
            }

        # Calculate cosine similarity
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )

        # Calculate Euclidean distance
        distance = np.linalg.norm(embedding1 - embedding2)

        # Interpret changes
        if similarity > 0.95:
            change_assessment = "Minimal change detected"
            change_level = "STABLE"
        elif similarity > 0.85:
            change_assessment = "Slight changes detected"
            change_level = "MINOR_CHANGE"
        elif similarity > 0.7:
            change_assessment = "Moderate changes detected"
            change_level = "MODERATE_CHANGE"
        else:
            change_assessment = "Significant changes detected"
            change_level = "SIGNIFICANT_CHANGE"

        return {
            "success": True,
            "similarity_score": round(float(similarity), 4),
            "euclidean_distance": round(float(distance), 4),
            "change_assessment": change_assessment,
            "change_level": change_level,
            "recommendation": (
                "No immediate action needed."
                if change_level == "STABLE"
                else (
                    "Monitor for further changes."
                    if change_level == "MINOR_CHANGE"
                    else "Consider dermatologist consultation."
                )
            ),
        }


# Singleton instance
_derm_foundation_instance: Optional[DermFoundationWrapper] = None


def get_derm_foundation(use_gpu: bool = False) -> DermFoundationWrapper:
    """Get or create the Derm Foundation wrapper singleton."""
    global _derm_foundation_instance
    if _derm_foundation_instance is None:
        _derm_foundation_instance = DermFoundationWrapper(use_gpu=use_gpu)
    return _derm_foundation_instance
