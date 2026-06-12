"""
모델 로드 및 추론 유틸리티
FastAPI 엔드포인트가 이 모듈을 import하여 사용합니다.
"""

import torch
import torch.nn as nn
from torchvision import transforms


# ===== 모델 정의 =====
class SimpleClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ===== 전처리 =====
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


# ===== 모델 로드 =====
def load_model(model_path: str) -> nn.Module:
    """저장된 state_dict에서 모델을 로드합니다."""
    model = SimpleClassifier(num_classes=10)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


# ===== 추론 =====
def predict(model: nn.Module, input_tensor: torch.Tensor) -> dict:
    """
    모델에 입력을 전달하고 결과를 반환합니다.

    Returns:
        dict: {
            "predicted_class": int,
            "confidence": float,
            "probabilities": dict[str, float]
        }
    """
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        confidence, predicted = probabilities.max(dim=1)

    probs_dict = {str(i): round(p.item(), 4) for i, p in enumerate(probabilities[0])}

    return {
        "predicted_class": predicted.item(),
        "confidence": round(confidence.item(), 4),
        "probabilities": probs_dict,
    }
