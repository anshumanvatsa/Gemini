"""
Vision Engine — PreViral
Analyzes uploaded image/video thumbnail to produce 7 vision features:
face_count, face_prominence_score, text_density, brightness_score,
color_vibrancy, clip_semantic_score, scene_cut_count
"""
import io
import os
import numpy as np
from PIL import Image
import cv2

# Lazy-load CLIP
_clip_model = None
_clip_processor = None

def _get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPProcessor, CLIPModel
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor

# Platform-specific ideal content descriptions for CLIP scoring
PLATFORM_IDEALS = {
    "instagram": "vibrant aesthetic lifestyle photo with clear subject and warm colors",
    "tiktok": "dynamic energetic person talking to camera with text overlay and bright background",
    "youtube": "clear thumbnail with bold text and human face expressing emotion",
    "twitter": "clear informative image or chart with high contrast",
    "linkedin": "professional business setting or clean infographic",
    "facebook": "warm social family or community gathering scene",
    "reddit": "interesting or funny image with clear subject matter",
}

def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Load image bytes into OpenCV format."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def analyze_faces(img: np.ndarray) -> tuple:
    """Detect faces and return (count, prominence_score)."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            return 0, 0.0

        h, w = img.shape[:2]
        img_area = h * w

        # Prominence = largest face area / total image area
        face_areas = [fw * fh for (_, _, fw, fh) in faces]
        max_face_area = max(face_areas)
        prominence = min(1.0, max_face_area / img_area * 4)  # Scale up so 25% of frame = 1.0

        return len(faces), round(prominence, 3)
    except Exception:
        return 0, 0.0

def analyze_brightness_vibrancy(img: np.ndarray) -> tuple:
    """Returns (brightness_score, color_vibrancy) from HSV histogram."""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        brightness = float(np.mean(v)) / 255.0  # 0-1
        vibrancy = float(np.mean(s)) / 255.0    # 0-1 (saturation = colorfulness)

        return round(brightness, 3), round(vibrancy, 3)
    except Exception:
        return 0.5, 0.5

def analyze_text_density(img: np.ndarray) -> float:
    """Estimate text density using edge detection (proxy for text pixels)."""
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Canny edge detection - text has many edges
        edges = cv2.Canny(gray, 100, 200)
        text_density = float(np.count_nonzero(edges)) / edges.size
        return round(min(1.0, text_density * 5), 3)  # Scale for typical range
    except Exception:
        return 0.0

def analyze_clip_score(image_bytes: bytes, platform: str) -> float:
    """
    Use CLIP to score how well the image matches the platform's ideal content.
    Returns a similarity score 0-1.
    """
    try:
        model, processor = _get_clip()
        import torch

        ideal_text = PLATFORM_IDEALS.get(platform.lower(), "engaging social media post")
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        inputs = processor(
            text=[ideal_text],
            images=pil_image,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits_per_image
            # Convert logit to 0-1 score using sigmoid
            score = float(torch.sigmoid(logits / 10.0).squeeze())

        return round(min(1.0, max(0.0, score)), 3)
    except Exception:
        return 0.5  # Neutral fallback

def analyze_image(image_bytes: bytes, platform: str) -> dict:
    """
    Main vision analysis function. Returns the 7-feature dict.
    """
    try:
        img = load_image_from_bytes(image_bytes)
        if img is None:
            raise ValueError("Could not decode image")

        face_count, face_prominence = analyze_faces(img)
        brightness, vibrancy = analyze_brightness_vibrancy(img)
        text_density = analyze_text_density(img)
        clip_score = analyze_clip_score(image_bytes, platform)

        return {
            "face_count": face_count,
            "face_prominence_score": face_prominence,
            "text_density": text_density,
            "brightness_score": brightness,
            "color_vibrancy": vibrancy,
            "clip_semantic_score": clip_score,
            "scene_cut_count": 0  # For images, always 0. Populated for videos separately.
        }
    except Exception as e:
        print(f"Vision engine error: {e}")
        # Return neutral defaults if vision fails - don't crash the whole prediction
        return {
            "face_count": 0,
            "face_prominence_score": 0.0,
            "text_density": 0.0,
            "brightness_score": 0.5,
            "color_vibrancy": 0.5,
            "clip_semantic_score": 0.5,
            "scene_cut_count": 0
        }

def no_image_defaults() -> dict:
    """Return defaults when no image is uploaded."""
    return {
        "face_count": 0,
        "face_prominence_score": 0.0,
        "text_density": 0.0,
        "brightness_score": 0.5,
        "color_vibrancy": 0.5,
        "clip_semantic_score": 0.5,
        "scene_cut_count": 0
    }


if __name__ == "__main__":
    # Test with a sample image from picsum
    import httpx
    print("Downloading test image...")
    r = httpx.get("https://picsum.photos/640/480")
    result = analyze_image(r.content, "instagram")
    print("Vision Analysis Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
