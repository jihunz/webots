from controller import Robot
from openai import OpenAI
import os
import dotenv
import json
import threading
import time
from queue import Queue


# =====================================================
#  로봇 동작 함수
# =====================================================

def move_joints(targets, speed=1.0, duration=3.0):
    """주어진 조인트 각도로 이동 (dict 또는 list 모두 허용)"""
    # 리스트 형태라면 dict으로 변환
    if isinstance(targets, list):
        converted = {}
        for t in targets:
            if isinstance(t, dict):
                joint = t.get("joint")
                angle = t.get("angle")
                if joint is not None and angle is not None:
                    converted[joint] = angle
        targets = converted

    elif not isinstance(targets, dict):
        print("⚠️ move_joints(): targets 형식이 잘못되었습니다.", type(targets))
        return

    for name, angle in targets.items():
        m = motors.get(name)
        if not m:
            print(f"⚠️ 모터 '{name}' 없음, 무시")
            continue
        m.setVelocity(abs(speed))
        m.setPosition(angle)

    steps = int(duration * 1000 / robot.getBasicTimeStep())
    for _ in range(steps):
        if robot.step(timestep) == -1:
            break
    print(f"🦾 조인트 이동 완료 → {targets}")


def open_gripper(speed=0.5, duration=2.0):
    """그리퍼 열기"""
    for name in GRIPPER_NAMES:
        motors[name].setVelocity(-abs(speed))
    steps = int(duration * 1000 / robot.getBasicTimeStep())
    for _ in range(steps):
        if robot.step(timestep) == -1:
            break
    for m in motors.values():
        m.setVelocity(0.0)
    print("✅ 그리퍼 열림")


def close_gripper(speed=0.5, duration=2.0):
    """그리퍼 닫기"""
    for name in GRIPPER_NAMES:
        motors[name].setVelocity(abs(speed))
    steps = int(duration * 1000 / robot.getBasicTimeStep())
    for _ in range(steps):
        if robot.step(timestep) == -1:
            break
    for m in motors.values():
        m.setVelocity(0.0)
    print("✅ 그리퍼 닫힘")


# =====================================================
#  명령 큐 / 스레드
# =====================================================

command_queue = Queue()
is_executing = False


def execute_command_queue():
    """큐에 쌓인 명령 순차 실행"""
    global is_executing
    while True:
        if not command_queue.empty():
            is_executing = True
            cmd = command_queue.get()
            try:
                t = cmd.get("type")
                if t == "move_joints":
                    move_joints(cmd.get("targets", {}), cmd.get("speed", 1.0), cmd.get("duration", 3.0))
                elif t == "open_gripper":
                    open_gripper(cmd.get("speed", 0.5), cmd.get("duration", 2.0))
                elif t == "close_gripper":
                    close_gripper(cmd.get("speed", 0.5), cmd.get("duration", 2.0))
            except Exception as e:
                print(f"명령 오류: {e}")
            finally:
                command_queue.task_done()
                is_executing = False
        else:
            time.sleep(0.1)


# =====================================================
#  LLM Function 정의
# =====================================================

functions = [
    {
        "name": "move_arm",
        "description": "UR10e의 팔 조인트를 제어합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "조인트 이름 → 라디안 각도"
                },
                "speed": {"type": "number", "default": 1.0},
                "duration": {"type": "number", "default": 3.0}
            },
            "required": []
        }
    },
    {
        "name": "control_gripper",
        "description": "UR10e 그리퍼를 열거나 닫습니다.",
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


# =====================================================
#  Function 처리 로직
# =====================================================

def process_function_call(fn_name, args):
    """LLM 함수 호출 → 실제 큐 명령으로 변환"""
    if fn_name == "move_arm":
        raw_targets = args.get("targets")

        # 리스트로 왔으면 dict으로 변환
        if isinstance(raw_targets, list):
            temp = {}
            for t in raw_targets:
                if isinstance(t, dict) and "joint" in t and "angle" in t:
                    temp[t["joint"]] = t["angle"]
            targets = temp
        else:
            targets = raw_targets or {"shoulder_lift_joint": -1.0, "elbow_joint": 1.5}

        cmd = {
            "type": "move_joints",
            "targets": targets,
            "speed": args.get("speed", 1.0),
            "duration": args.get("duration", 3.0)
        }
        command_queue.put(cmd)
        return f"팔 이동 명령 추가됨 → {targets}"

    elif fn_name == "control_gripper":
        act = args.get("action")
        cmd_type = "open_gripper" if act == "open" else "close_gripper"
        command_queue.put({
            "type": cmd_type,
            "speed": args.get("speed", 0.5),
            "duration": args.get("duration", 2.0)
        })
        return f"그리퍼 {act} 명령 추가됨"

    return "알 수 없는 함수"


# =====================================================
#  LLM 처리
# =====================================================

def handle_llm_command(user_message):
    """자연어 명령을 LLM Function Calling으로 처리"""
    if client is None:
        return "❌ OpenAI 클라이언트 없음"

    messages = [
        {
            "role": "system",
            "content": (
                "너는 UR10e 산업용 로봇팔 제어 에이전트야. "
                "자연어 명령을 해석해 move_arm 또는 control_gripper 함수를 호출해야 해. "
                "만약 사용자가 단순히 '팔을 들어올려'라고 말하면, "
                "shoulder_lift_joint: -1.0, elbow_joint: 1.5 로 설정해."
            )
        },
        {"role": "user", "content": user_message}
    ]

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
            return process_function_call(fn, args)
        return msg.content or "명령을 인식하지 못했습니다."
    except Exception as e:
        return f"LLM 오류: {e}"


# =====================================================
#  초기화
# =====================================================

dotenv.load_dotenv()
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("✅ OpenAI 연결 완료")
except Exception as e:
    print(f"❌ OpenAI 연결 실패: {e}")
    client = None

robot = Robot()
timestep = int(robot.getBasicTimeStep())

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
    try:
        m = robot.getDevice(name)
        if name in GRIPPER_NAMES:
            m.setPosition(float('inf'))  # velocity mode
            m.setVelocity(0.0)
        else:
            m.setVelocity(1.0)
        motors[name] = m
    except Exception as e:
        print(f"⚠️ 모터 {name} 초기화 실패: {e}")

print("✅ 로드된 모터:", list(motors.keys()))

# 명령 스레드 시작
threading.Thread(target=execute_command_queue, daemon=True).start()
print("🚀 명령 큐 실행 스레드 시작됨")


# =====================================================
#  Webots RobotWindow (WWI) 인터페이스 루프
# =====================================================

while robot.step(timestep) != -1:
    message = robot.wwiReceiveText()
    if message:
        print(f"📩 USER: {message}")
        result = handle_llm_command(message)
        print("🧠 처리 결과:", result)
        robot.wwiSendText(f"<b>{message}</b><br>{result}")
