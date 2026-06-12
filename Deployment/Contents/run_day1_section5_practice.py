"""
모델 배포 개론 Day 1 - 섹션 5: 종합 실습
MNIST 모델 학습 → 직렬화 → 검증 → API 연결 준비
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np

print("=" * 60)
print("섹션 5: 종합 실습 — MNIST 모델 배포 파이프라인")
print("=" * 60)


# ===== Step 1: 모델 정의 =====
print("\n" + "=" * 50)
print("Step 1: 모델 정의")
print("=" * 50)

class SimpleClassifier(nn.Module):
    """
    간단한 이미지 분류 모델
    - 입력: 1x28x28 (MNIST)
    - 출력: 10개 클래스에 대한 확률
    """
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


# ===== Step 2: 데이터 준비 및 학습 =====
print("\n" + "=" * 50)
print("Step 2: 데이터 준비 및 모델 학습")
print("=" * 50)

# 데이터 변환 정의
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST 데이터 로드
train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print(f"학습 데이터: {len(train_dataset)}장")
print(f"테스트 데이터: {len(test_dataset)}장")

# 모델, 손실 함수, 옵티마이저
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleClassifier(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 (3 에포크)
num_epochs = 3

for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # 200 배치마다 진행 상황 출력
        if (batch_idx + 1) % 200 == 0:
            print(f"  Epoch {epoch} [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {running_loss/(batch_idx+1):.4f} "
                  f"Acc: {100.*correct/total:.1f}%")

    # 에포크 종료 후 테스트
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_total += labels.size(0)
            test_correct += predicted.eq(labels).sum().item()

    print(f"  → Epoch {epoch} 완료 | 테스트 정확도: {100.*test_correct/test_total:.1f}%")

print(f"\n🎉 학습 완료! 최종 테스트 정확도: {100.*test_correct/test_total:.1f}%")


# ===== Step 3: 모델 직렬화 (3가지 방식) =====
print("\n" + "=" * 50)
print("Step 3: 모델 직렬화")
print("=" * 50)

os.makedirs("models", exist_ok=True)

# 모델을 CPU로 이동
model_cpu = model.cpu()
model_cpu.eval()

# 추론 비교용 테스트 입력
test_input = test_dataset[0][0].unsqueeze(0)
test_label = test_dataset[0][1]

print(f"테스트 입력 크기: {test_input.shape}")
print(f"정답 레이블: {test_label}")

# 원본 모델의 추론 결과 기록
with torch.no_grad():
    original_output = model_cpu(test_input)
    original_pred = original_output.argmax(dim=1).item()
    original_conf = torch.softmax(original_output, dim=1).max().item()

print(f"\n원본 모델 예측: {original_pred} (확신도: {original_conf:.4f})")
print(f"정답:          {test_label}")
print(f"정답 여부:      {'✅ 맞음' if original_pred == test_label else '❌ 틀림'}")

# --- 3-1: state_dict 저장 ---
print("\n--- state_dict 저장 ---")
torch.save(model_cpu.state_dict(), "models/mnist_state_dict.pth")
print(f"✅ state_dict 저장 완료: {os.path.getsize('models/mnist_state_dict.pth') / 1024:.1f} KB")

# --- 3-2: TorchScript 저장 ---
print("\n--- TorchScript 저장 ---")
traced_model = torch.jit.trace(model_cpu, test_input)
traced_model.save("models/mnist_traced.pt")
print(f"✅ TorchScript 저장 완료: {os.path.getsize('models/mnist_traced.pt') / 1024:.1f} KB")

# --- 3-3: ONNX 저장 ---
print("\n--- ONNX 저장 ---")
torch.onnx.export(
    model_cpu,
    test_input,
    "models/mnist_model.onnx",
    export_params=True,
    opset_version=17,
    input_names=["image"],
    output_names=["prediction"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "prediction": {0: "batch_size"},
    }
)
print(f"✅ ONNX 저장 완료: {os.path.getsize('models/mnist_model.onnx') / 1024:.1f} KB")

# 저장 결과 요약
print("\n" + "=" * 50)
print("📁 models/ 폴더 내용")
print("=" * 50)
for fname in sorted(os.listdir("models")):
    fpath = os.path.join("models", fname)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {fname:<30} {size_kb:>8.1f} KB")


# ===== Step 4: 불러오기 및 추론 검증 =====
print("\n" + "=" * 50)
print("Step 4: 불러오기 및 추론 검증")
print("=" * 50)

# 검증 1: state_dict
print("\n--- 검증 1: state_dict ---")
loaded_sd = SimpleClassifier(num_classes=10)
loaded_sd.load_state_dict(
    torch.load("models/mnist_state_dict.pth", weights_only=True)
)
loaded_sd.eval()

with torch.no_grad():
    sd_output = loaded_sd(test_input)
    sd_pred = sd_output.argmax(dim=1).item()

print(f"[state_dict] 예측: {sd_pred}, 원본과 일치: {torch.allclose(original_output, sd_output)}")

# 검증 2: TorchScript
print("\n--- 검증 2: TorchScript ---")
loaded_ts = torch.jit.load("models/mnist_traced.pt")

with torch.no_grad():
    ts_output = loaded_ts(test_input)
    ts_pred = ts_output.argmax(dim=1).item()

print(f"[TorchScript] 예측: {ts_pred}, 원본과 일치: {torch.allclose(original_output, ts_output)}")

# 검증 3: ONNX
print("\n--- 검증 3: ONNX ---")
import onnxruntime as ort

session = ort.InferenceSession("models/mnist_model.onnx")
onnx_output = session.run(
    ["prediction"],
    {"image": test_input.numpy()}
)

onnx_pred = np.argmax(onnx_output[0], axis=1)[0]
match = np.allclose(original_output.numpy(), onnx_output[0], atol=1e-5)

print(f"[ONNX]        예측: {onnx_pred}, 원본과 일치 (오차 허용): {match}")

# 종합 검증 결과
print("\n" + "=" * 60)
print("📊 직렬화 검증 결과 요약")
print("=" * 60)
print(f"  정답 레이블:        {test_label}")
print(f"  원본 모델 예측:     {original_pred}")
print(f"  state_dict 예측:   {sd_pred}  {'✅' if sd_pred == original_pred else '❌'}")
print(f"  TorchScript 예측:  {ts_pred}  {'✅' if ts_pred == original_pred else '❌'}")
print(f"  ONNX 예측:         {onnx_pred}  {'✅' if onnx_pred == original_pred else '❌'}")
print("=" * 60)

if all(p == original_pred for p in [sd_pred, ts_pred, onnx_pred]):
    print("\n🎉 세 가지 방식 모두 원본과 동일한 결과를 반환합니다.")
    print("   모델을 안전하게 직렬화하고 복원할 수 있다는 것이 검증되었습니다.")


# ===== Step 5: 배치 추론 테스트 =====
print("\n" + "=" * 50)
print("Step 5: 배치 추론 테스트")
print("=" * 50)

batch_images = torch.stack([test_dataset[i][0] for i in range(8)])
batch_labels = [test_dataset[i][1] for i in range(8)]

print(f"배치 입력 크기: {batch_images.shape}")

with torch.no_grad():
    sd_batch = loaded_sd(batch_images).argmax(dim=1).tolist()
    ts_batch = loaded_ts(batch_images).argmax(dim=1).tolist()

onnx_batch_out = session.run(["prediction"], {"image": batch_images.numpy()})
onnx_batch = np.argmax(onnx_batch_out[0], axis=1).tolist()

print(f"\n{'이미지':<8} {'정답':<6} {'state_dict':<12} {'TorchScript':<13} {'ONNX':<8}")
print("-" * 50)
for i in range(8):
    match_emoji = "✅" if sd_batch[i] == ts_batch[i] == onnx_batch[i] == batch_labels[i] else "❌"
    print(f"  #{i:<5} {batch_labels[i]:<6} {sd_batch[i]:<12} {ts_batch[i]:<13} {onnx_batch[i]:<8} {match_emoji}")


# ===== Step 6: API 연결 준비 — 추론 함수 분리 =====
print("\n" + "=" * 50)
print("Step 6: API 연결 준비 — 추론 함수 분리")
print("=" * 50)

# app 디렉토리 생성
os.makedirs("app", exist_ok=True)

# model_utils.py 작성
model_utils_code = '''"""
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
'''

with open("app/model_utils.py", "w") as f:
    f.write(model_utils_code)
print("✅ app/model_utils.py 작성 완료")

# app/__init__.py 생성
with open("app/__init__.py", "w") as f:
    pass
print("✅ app/__init__.py 작성 완료")

# model_utils.py 테스트
import sys
sys.path.insert(0, ".")

from app.model_utils import load_model, predict

model_for_api = load_model("models/mnist_state_dict.pth")
result = predict(model_for_api, test_input)

print("\n추론 결과:")
print(f"  예측 클래스: {result['predicted_class']}")
print(f"  확신도:     {result['confidence']}")
print(f"  전체 확률:")
for cls, prob in result['probabilities'].items():
    bar = "█" * int(prob * 50)
    print(f"    {cls}: {prob:.4f} {bar}")


# ===== Step 7: 프로젝트 최종 구조 확인 =====
print("\n" + "=" * 50)
print("Step 7: 프로젝트 최종 구조 확인")
print("=" * 50)

def show_tree(path, prefix="", max_depth=2, current_depth=0):
    """프로젝트 폴더 구조를 트리 형태로 출력합니다."""
    if current_depth >= max_depth:
        return

    entries = sorted(os.listdir(path))
    # .venv 등 무거운 폴더는 제외
    exclude = {".venv", ".venv_hckim", ".venv_test", "__pycache__", ".ipynb_checkpoints", "data"}
    entries = [e for e in entries if e not in exclude]

    for i, entry in enumerate(entries):
        full_path = os.path.join(path, entry)
        connector = "└── " if i == len(entries) - 1 else "├── "

        if os.path.isdir(full_path):
            print(f"{prefix}{connector}📁 {entry}/")
            extension = "    " if i == len(entries) - 1 else "│   "
            show_tree(full_path, prefix + extension, max_depth, current_depth + 1)
        else:
            size = os.path.getsize(full_path) / 1024
            print(f"{prefix}{connector}{entry} ({size:.1f} KB)")

show_tree(".")

print("\n" + "=" * 60)
print("🎉 Day 1 모든 실습이 완료되었습니다!")
print("=" * 60)
