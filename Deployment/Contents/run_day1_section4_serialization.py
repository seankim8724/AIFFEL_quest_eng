"""
모델 배포 개론 Day 1 - 섹션 4: 모델 직렬화 기초
실습 코드 (노트북 셀 기반)
"""
import torch
import torch.nn as nn
import os
import numpy as np

print("=" * 60)
print("섹션 4: 모델 직렬화 — state_dict, TorchScript, ONNX")
print("=" * 60)

# ===== 모델 정의 =====
class SimpleClassifier(nn.Module):
    """
    간단한 이미지 분류 모델
    - 입력: 1x28x28 (MNIST와 동일한 크기)
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


# ===== 모델 생성 및 테스트 =====
print("\n--- 모델 생성 ---")
model = SimpleClassifier(num_classes=10)
dummy_input = torch.randn(1, 1, 28, 28)
output = model(dummy_input)

print(f"모델 구조:\n{model}\n")
print(f"입력 크기: {dummy_input.shape}")
print(f"출력 크기: {output.shape}")
print(f"출력 값:   {output.detach()}")

# ===== 모델 저장 폴더 생성 =====
os.makedirs("models", exist_ok=True)

# ============================================
# 방법 1: state_dict (가중치만 저장)
# ============================================
print("\n" + "=" * 50)
print("방법 1: state_dict")
print("=" * 50)

model.eval()
with torch.no_grad():
    output = model(dummy_input)

torch.save(model.state_dict(), "models/model_state_dict.pth")

file_size = os.path.getsize("models/model_state_dict.pth")
print(f"저장 완료: models/model_state_dict.pth")
print(f"파일 크기: {file_size / 1024:.1f} KB")

# 저장된 키 확인
state_dict = torch.load("models/model_state_dict.pth", weights_only=True)
print("\n저장된 키 목록:")
for key, tensor in state_dict.items():
    print(f"  {key:40s} → {tensor.shape}")

# 불러오기 및 검증
loaded_model = SimpleClassifier(num_classes=10)
loaded_model.load_state_dict(torch.load("models/model_state_dict.pth", weights_only=True))
loaded_model.eval()

with torch.no_grad():
    loaded_output = loaded_model(dummy_input)

print(f"\n원본 출력:  {output.detach()}")
print(f"복원 출력:  {loaded_output}")
print(f"동일 여부:  {torch.allclose(output.detach(), loaded_output)}")

# ============================================
# 방법 2: TorchScript
# ============================================
print("\n" + "=" * 50)
print("방법 2: TorchScript")
print("=" * 50)

model.eval()
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("models/model_traced.pt")

file_size = os.path.getsize("models/model_traced.pt")
print(f"저장 완료: models/model_traced.pt")
print(f"파일 크기: {file_size / 1024:.1f} KB")

# 불러오기 (클래스 정의 불필요!)
loaded_traced = torch.jit.load("models/model_traced.pt")

with torch.no_grad():
    traced_output = loaded_traced(dummy_input)

print(f"\n원본 출력:       {output.detach()}")
print(f"TorchScript 출력: {traced_output}")
print(f"동일 여부:        {torch.allclose(output.detach(), traced_output)}")

# trace vs script 비교
traced = torch.jit.trace(model, dummy_input)
scripted = torch.jit.script(model)

with torch.no_grad():
    trace_out = traced(dummy_input)
    script_out = scripted(dummy_input)

print(f"\ntrace 출력:  {trace_out}")
print(f"script 출력: {script_out}")
print(f"동일 여부:   {torch.allclose(trace_out, script_out)}")

# ============================================
# 방법 3: ONNX
# ============================================
print("\n" + "=" * 50)
print("방법 3: ONNX")
print("=" * 50)

import onnx

model.eval()
torch.onnx.export(
    model,
    dummy_input,
    "models/model.onnx",
    export_params=True,
    opset_version=17,
    input_names=["image"],
    output_names=["prediction"],
    dynamic_axes={
        "image": {0: "batch_size"},
        "prediction": {0: "batch_size"},
    }
)

file_size = os.path.getsize("models/model.onnx")
print(f"저장 완료: models/model.onnx")
print(f"파일 크기: {file_size / 1024:.1f} KB")

# ONNX 모델 검증
onnx_model = onnx.load("models/model.onnx")
onnx.checker.check_model(onnx_model)
print("✅ ONNX 모델 검증 통과")

print(f"\n입력:")
for inp in onnx_model.graph.input:
    print(f"  이름: {inp.name}")
    shape = [d.dim_param or d.dim_value for d in inp.type.tensor_type.shape.dim]
    print(f"  크기: {shape}")

print(f"\n출력:")
for out in onnx_model.graph.output:
    print(f"  이름: {out.name}")
    shape = [d.dim_param or d.dim_value for d in out.type.tensor_type.shape.dim]
    print(f"  크기: {shape}")

# ONNX Runtime으로 추론
import onnxruntime as ort

session = ort.InferenceSession("models/model.onnx")
input_data = dummy_input.numpy()

onnx_output = session.run(
    output_names=["prediction"],
    input_feed={"image": input_data}
)

print(f"\nPyTorch 출력:       {output.detach().numpy()}")
print(f"ONNX Runtime 출력:  {onnx_output[0]}")
print(f"동일 여부 (오차 허용): {np.allclose(output.detach().numpy(), onnx_output[0], atol=1e-5)}")

print("\n✅ 섹션 4 완료!")
