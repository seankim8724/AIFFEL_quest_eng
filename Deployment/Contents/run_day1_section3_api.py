"""
모델 배포 개론 Day 1 - 섹션 3: RESTful API 기초
실습 코드 (노트북 셀 기반)
"""
import json
import requests

print("=" * 60)
print("섹션 3: RESTful API 기초 — HTTP 요청 실습")
print("=" * 60)

# ===== GET 요청 =====
print("\n--- GET 요청 ---")
response = requests.get("https://jsonplaceholder.typicode.com/posts/1")
print(f"상태 코드: {response.status_code}")
print(f"응답 내용:")
print(json.dumps(response.json(), ensure_ascii=False, indent=2))

# ===== POST 요청 =====
print("\n--- POST 요청 ---")
new_post = {
    "title": "AI 모델 배포하기",
    "body": "FastAPI를 사용한 모델 배포를 학습합니다.",
    "userId": 1
}
response = requests.post(
    "https://jsonplaceholder.typicode.com/posts",
    json=new_post
)
print(f"상태 코드: {response.status_code}")
print(f"응답 내용:")
print(json.dumps(response.json(), ensure_ascii=False, indent=2))

# ===== 404 에러 =====
print("\n--- 존재하지 않는 리소스 요청 ---")
response = requests.get("https://jsonplaceholder.typicode.com/posts/99999")
print(f"상태 코드: {response.status_code}")
print(f"응답 내용: {response.json()}")

# ===== 잘못된 URL =====
print("\n--- 잘못된 URL 요청 ---")
try:
    response = requests.get("https://jsonplaceholder.typicode.com/없는경로")
    print(f"상태 코드: {response.status_code}")
except requests.exceptions.RequestException as e:
    print(f"요청 실패: {e}")

print("\n✅ 섹션 3 완료!")
