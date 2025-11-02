from controller import Robot
from openai import OpenAI
import os
import time
import dotenv
import json
import threading
from queue import Queue

# =====================================================
#  조인트 및 그리퍼 제어 함수
# =====================================================

def move_joints(targets: dict, speed=1.0, duration=3.0):
    """주어진 조인트 각도로 이동"""
    for name, angle in targets.items():
        m = motors.get(name)
        if not m:
            continue
        m.setVelocity(abs(speed))
        m.setPosition(angle)
    steps = int(duration * 1000 / robot.getBasicTimeStep())
    for _ in range(steps):
        if robot.step(timestep) == -1:
            break
    print(f"조인트 이동 완료 → {targets}")


def open_gripper(speed=0.5, duration=2.0):
    """3-finger 그리퍼 열기 (velocity-mode)"""
    for name in GRIPPER_NAMES:
        m = motors[name]
        m.setVelocity(-abs(speed))
    steps = int(duration * 1000 / robot.getBasicTimeStep())
    for _ in range(steps):
        if robot.step(timestep) == -1:
            break
    for m in motors.values():
        m.setVelocity(0.0)
    print("✅ 그리퍼 열림")


def close_gripper(speed=0.5, duration=2.0):
    """3-finger 그리퍼 닫기 (velocity-mode)"""
    for name in GRIPPER_NAMES:
        m = motors[name]
        m.setVelocity(abs(speed))
    steps = int(duration * 1000 / robot.getBasicTimeStep())
    for _ in range(steps):
        if robot.step(timestep) == -1:
            break
    for m in motors.values():
        m.setVelocity(0.0)
    print("✅ 그리퍼 닫힘")


# =====================================================
#  명령 큐 / 실행 스레드
# =====================================================

command_queue = Queue()
is_executing = False

def execute_command_queue():
    """큐에 쌓인 명령을 순차적으로 실행"""
    global is_executing
    while True:
        if not command_queue.empty():
            is_executing = True
            cmd = command_queue.get()
            try:
                cmd_type = cmd.get("type")
                if cmd_type == "move_joints":
                    move_joints(cmd["targets"], cmd.get("speed",1.0), cmd.get("duration",3.0))
                elif cmd_type == "open_gripper":
                    open_gripper(cmd.get("speed",0.5), cmd.get("duration",2.0))
                elif cmd_type == "close_gripper":
                    close_gripper(cmd.get("speed",0.5), cmd.get("duration",2.0))
            except Exception as e:
                print(f"명령 실행 오류: {e}")
            finally:
                command_queue.task_done()
                is_executing = False
        else:
            time.sleep(0.1)


# =====================================================
#  Function-Calling 스키마 정의
# =====================================================

functions = [
    {
        "name": "move_arm",
        "description": "UR10e 팔의 여러 조인트를 움직입니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "object",
                    "description": "각 조인트 이름 → 라디안 값",
                    "additionalProperties": {"type": "number"}
                },
                "speed": {"type": "number", "default": 1.0},
                "duration": {"type": "number", "default": 3.0}
            },
            "required": ["targets"]
        }
    },
    {
        "name": "control_gripper",
        "description": "3-Finger 그리퍼를 열거나 닫습니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["open", "close"]},
                "speed": {"type": "number", "default": 0.5},
                "duration": {"type": "number", "default": 2.0}
            },
            "required": ["action"]
        }
    }
]


def process_function_call(function_name, arguments):
    """LLM 함수 호출 → 큐 적재"""
    if function_name == "move_arm":
        cmd = {
            "type": "move_joints",
            "targets": arguments.get("targets", {}),
            "speed": arguments.get("speed", 1.0),
            "duration": arguments.get("duration", 3.0)
        }
        command_queue.put(cmd)
        return f"팔 이동 명령 추가 ({len(cmd['targets'])} joints)."
    elif function_name == "control_gripper":
        act = arguments.get("action")
        cmd = {"type": "open_gripper" if act=="open" else "close_gripper",
               "speed": arguments.get("speed",0.5),
               "duration": arguments.get("duration",2.0)}
        command_queue.put(cmd)
        return f"그리퍼 {act} 명령 추가."
    return "알 수 없는 함수 요청."


# =====================================================
#  LLM Function-Calling 래퍼
# =====================================================

def handle_llm_function_calling(user_message):
    if client is None:
        return "OpenAI 클라이언트 없음"
    messages = [
        {
            "role": "system",
            "content": (
                "너는 UR10e 산업용 로봇팔 제어 에이전트야.\n"
                "사용자의 자연어 명령을 해석해 move_arm 또는 control_gripper 함수를 호출해야 해.\n\n"
                "예시:\n"
                " - '팔을 들어올려' → move_arm(targets={'shoulder_lift_joint': -1.2})\n"
                " - '그리퍼를 열어' → control_gripper(action='open')"
            )
        },
        {"role": "user", "content": user_message}
    ]
    print(f"LLM 요청: {user_message}")

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            functions=functions,
            function_call="auto",
            max_completion_tokens=256,
            timeout=20
        )
        msg = resp.choices[0].message
        if hasattr(msg, "function_call") and msg.function_call:
            fn = msg.function_call.name
            args = json.loads(msg.function_call.arguments)
            result = process_function_call(fn, args)
            return result
        return msg.content or "함수 호출 없음"
    except Exception as e:
        return f"LLM 오류: {e}"


# =====================================================
#  초기화
# =====================================================

dotenv.load_dotenv()
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ OpenAI 클라이언트 초기화 완료")
except Exception as e:
    print(f"❌ OpenAI 초기화 실패: {e}")
    client = None

robot = Robot()
timestep = int(robot.getBasicTimeStep())

# UR10e 조인트 및 그리퍼 초기화
JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint"
]
GRIPPER_NAMES = [
    "finger_1_joint_1",
    "finger_2_joint_1",
    "finger_middle_joint_1"
]

motors = {}
for name in JOINT_NAMES + GRIPPER_NAMES:
    m = robot.getDevice(name)
    if name in GRIPPER_NAMES:
        m.setPosition(float('inf'))  # velocity-mode
        m.setVelocity(0.0)
    else:
        m.setVelocity(1.0)
    motors[name] = m
print(f"로드된 모터: {list(motors.keys())}")

# 큐 스레드 시작
threading.Thread(target=execute_command_queue, daemon=True).start()

# =====================================================
#  메인 루프
# =====================================================

print("🚀 UR10e LLM 제어 시작")

while robot.step(timestep) != -1:
    msg = robot.wwiReceiveText()
    if msg:
        print("USER_MESSAGE:", msg)
        result = handle_llm_function_calling(msg)
        reply = f"결과: {result}\n큐 크기: {command_queue.qsize()}"
        robot.wwiSendText(reply)
