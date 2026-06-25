import re

file_path = "/home/jimmy_linux/anaconda_projects/nlp/Social media emotion analyzer GitHub/src/predict.py"
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# Replace model_artifact_exists
target_exists = """def model_artifact_exists(info: ModelInfo) -> bool:
    \"\"\"Check whether the selected model artifact is available locally.\"\"\"

    if info.model_type == "transformer":
        return (info.path / "config.json").exists() and (
            (info.path / "model.safetensors").exists() or (info.path / "pytorch_model.bin").exists()
        )"""

replacement_exists = """def model_artifact_exists(info: ModelInfo) -> bool:
    \"\"\"Check whether the selected model artifact is available locally.\"\"\"

    if info.model_type == "transformer":
        return True"""

if target_exists in content:
    content = content.replace(target_exists, replacement_exists)
else:
    print("Could not find target_exists")

# Replace _predict_transformer
# Using regex to replace the entire function body since it's long
target_predict = re.compile(r'def _predict_transformer\(info: ModelInfo, text: str\) -> tuple\[str, dict\[str, float\]\]:.*?def predict_emotion\(', re.DOTALL)

replacement_predict = """def _predict_transformer(info: ModelInfo, text: str) -> tuple[str, dict[str, float]]:
    \"\"\"Predict emotion using a DistilBERT model downloaded dynamically.\"\"\"

    try:
        from transformers import pipeline
    except ImportError as exc:
        raise ImportError(
            "Install PyTorch and Transformers to use DistilBERT. "
            f"Original error: {exc}"
        ) from exc

    import functools
    @functools.lru_cache(maxsize=1)
    def _get_hf_pipeline():
        return pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)
        
    classifier = _get_hf_pipeline()
    
    results = classifier(text)[0]
    
    confidence = {label: 0.0 for label in LABEL_NAMES}
    for result in results:
        label = result["label"].lower()
        if label in confidence:
            confidence[label] = float(result["score"])

    prediction = max(confidence, key=confidence.get)
    return prediction, confidence

def predict_emotion("""

content = target_predict.sub(replacement_predict, content)

with open(file_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Updated predict.py")
