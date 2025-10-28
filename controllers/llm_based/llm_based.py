from controller import Supervisor
from openai import OpenAI
import os, time, dotenv

# ==================== 이동 제어 함수들 ====================

def move_stop(left_wheel, right_wheel):
    left_wheel.setPosition(0)
    right_wheel.setPosition(0)
    left_wheel.setVelocity(0)
    right_wheel.setVelocity(0)

def move_forward(left_wheel, right_wheel, speed=1.0):
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(float('inf'))
    left_wheel.setVelocity(speed)
    right_wheel.setVelocity(speed)

def move_backward(left_wheel, right_wheel, speed=1.0):
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(float('inf'))
    left_wheel.setVelocity(-speed)
    right_wheel.setVelocity(-speed)

def move_left(left_wheel, right_wheel, speed=1.0):
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(float('inf'))
    left_wheel.setVelocity(-speed)
    right_wheel.setVelocity(speed)

def move_right(left_wheel, right_wheel, speed=1.0):
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(float('inf'))
    left_wheel.setVelocity(speed)
    right_wheel.setVelocity(-speed)

# ==========================================================

def html_format(message: str) -> str:
    """HTML 포맷 정리 (필요시 로그 출력용)"""
    message = message.replace("<", "&lt;")
    message = message.replace(">", "&gt;")
    message = message.replace("\n", "<br>")
    return message

# ==========================================================
# OpenAI 클라이언트 초기화
# ==========================================================

dotenv.load_dotenv()  # .env 파일 로드

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ OpenAI 클라이언트 초기화 완료")
except Exception as e:
    print(f"❌ OpenAI 클라이언트 초기화 실패: {e}")
    print("API 키가 설정되지 않았거나 네트워크 연결 문제가 있을 수 있습니다.")
    client = None

# ==========================================================
# 자연어 → 제어 명령어 변환 함수
# ==========================================================

def parse_natural_language_with_llm(user_message: str) -> str:
    """
    LLM을 사용하여 자연어 명령을 파싱하고 제어 명령 문자열로 변환.
    실패 시 기본 명령어로 대체.
    """
    if client is None:
        print("⚠️ OpenAI 클라이언트가 없습니다. 기본 명령어를 사용합니다.")
        return "forward 1.0 1.0"

    try:
        system_prompt = """당신은 로봇 제어 시스템의 명령어 변환기입니다.
사용자의 자연어 명령을 받아서 다음 형식의 명령어로 변환해야 합니다:

사용 가능한 명령어:

forward [속도] [지속시간] : 앞으로 이동
backward [속도] [지속시간] : 뒤로 이동
left [속도] [지속시간] : 왼쪽으로 회전
right [속도] [지속시간] : 오른쪽으로 회전
stop [속도] [지속시간] : 정지

속도 범위: 0.1 ~ 2.5 (기본값: 1.0)
지속시간 범위: 0.1 ~ 10.0 초 (기본값: 1.0)

예시:
"앞으로 가줘" → "forward 1.0 1.0"
"뒤로 천천히 가줘" → "backward 0.5 1.0"
"왼쪽으로 빠르게 회전해줘" → "left 1.5 1.0"
"멈춰줘" → "stop 1.0 1.0"

명령어만 반환하고 다른 설명은 하지 마세요.
"""

        user_prompt = f"사용자 명령: {user_message}"

        response = client.chat.completions.create(
            model="gpt-5",  # ⚙️ 사용할 모델 (원하는 모델명으로 수정 가능)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1000,
            temperature=0.1,
            timeout=10
        )

        command = response.choices[0].message.content.strip()
        print(f"🤖 LLM 응답: {command}")

        return command

    except Exception as e:
        print(f"⚠️ LLM 파싱 오류: {e}")
        print("기본 명령어로 대체합니다.")
        return "forward 1.0 1.0"

# ==========================================================
# Webots Supervisor 초기화
# ==========================================================

robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
print(f"기본 시간 스텝: {timestep} ms")

# 장치 가져오기
left_wheel = robot.getDevice("MLW")
right_wheel = robot.getDevice("MRW")

# ==========================================================
# 예시: 사용자 명령을 LLM으로 해석 후 제어
# ==========================================================

user_input = "왼쪽으로 천천히 돌아줘"  # ← 테스트용 자연어 명령
command = parse_natural_language_with_llm(user_input)

try:
    action, speed, duration = command.split()
    speed = float(speed)
    duration = float(duration)
except Exception:
    print("⚠️ 명령어 파싱 실패 - 기본값 사용")
    action, speed, duration = "forward", 1.0, 1.0

print(f"🎯 실행 명령: {action}, 속도={speed}, 지속시간={duration}s")

# 명령 실행
if action == "forward":
    move_forward(left_wheel, right_wheel, speed)
elif action == "backward":
    move_backward(left_wheel, right_wheel, speed)
elif action == "left":
    move_left(left_wheel, right_wheel, speed)
elif action == "right":
    move_right(left_wheel, right_wheel, speed)
elif action == "stop":
    move_stop(left_wheel, right_wheel)

# 일정 시간 동안 동작 유지
end_time = robot.getTime() + duration
while robot.step(timestep) != -1:
    if robot.getTime() > end_time:
        move_stop(left_wheel, right_wheel)
        break
