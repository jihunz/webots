# controllers/ur10e_planner_controller/ur10e_planner_controller.py
from controller import Robot
from openai import OpenAI
import os
import dotenv
import json
import threading
import time
from queue import Queue

# ============================================
# 설정
# ============================================
dotenv.load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# ============================================
# 공통 유틸
# ============================================
def strip_code_fences(s: str) -> str:
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        # ```json ... ``` or ``` ... ```
        s = s.split("```", 2)
        if len(s) == 3:
            return s[1].split("\n", 1)[-1] if s[1].startswith(("json", "JSON")) else s[1]
    return s

def step_for(robot: Robot, timestep: int, duration_sec: float):
    end = time.time() + max(0.0, duration_sec)
    while time.time() < end:
        if robot.step(timestep) == -1:
            break

# ============================================
# 로봇 초기화
# ============================================
robot = Robot()
timestep = int(robot.getBasicTimeStep())

JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
GRIPPER_NAMES = [
    "finger_1_joint_1",
    "finger_2_joint_1",
    "finger_middle_joint_1",
]

motors = {}
for name in JOINT_NAMES + GRIPPER_NAMES:
    try:
        dev = robot.getDevice(name)
        if name in GRIPPER_NAMES:
            # gripper는 velocity-mode로 사용
            dev.setPosition(float('inf'))
            dev.setVelocity(0.0)
        else:
            dev.setVelocity(1.0)
        motors[name] = dev
    except Exception as e:
        print(f"[WARN] Device init failed: {name} ({e})")

print("✅ Loaded motors:", list(motors.keys()))

# ============================================
# 제어 함수 (조인트/그리퍼)
# ============================================
def move_joints(targets, speed=1.0, duration=3.0):
    """
    주어진 조인트 각도로 이동 (dict 또는 list 지원)
    - dict 예: {'shoulder_lift_joint': -1.0, 'elbow_joint': 1.5}
    - list 예: [{'joint':'shoulder_lift_joint','angle':-1.0}, ...]
    """
    # list -> dict 변환
    if isinstance(targets, list):
        converted = {}
        for item in targets:
            if isinstance(item, dict):
                j = item.get("joint")
                a = item.get("angle")
                if j is not None and a is not None:
                    converted[j] = float(a)
        targets = converted
    elif not isinstance(targets, dict):
        print("⚠️ move_joints(): invalid targets type:", type(targets))
        return

    # 적용
    for name, angle in targets.items():
        m = motors.get(name)
        if not m:
            print(f"⚠️ Unknown joint '{name}', skip")
            continue
        try:
            m.setVelocity(abs(speed))
            m.setPosition(float(angle))
        except Exception as e:
            print(f"⚠️ setPosition failed for {name}: {e}")

    step_for(robot, timestep, duration)
    print(f"🦾 Joints moved → {targets}")

def open_gripper(speed=0.5, duration=2.0):
    for name in GRIPPER_NAMES:
        m = motors.get(name)
        if m: m.setVelocity(-abs(speed))
    step_for(robot, timestep, duration)
    # stop velocity
    for name in GRIPPER_NAMES:
        m = motors.get(name)
        if m: m.setVelocity(0.0)
    print("✅ Gripper opened")

def close_gripper(speed=0.5, duration=2.0):
    for name in GRIPPER_NAMES:
        m = motors.get(name)
        if m: m.setVelocity(abs(speed))
    step_for(robot, timestep, duration)
    for name in GRIPPER_NAMES:
        m = motors.get(name)
        if m: m.setVelocity(0.0)
    print("✅ Gripper closed")

# ============================================
# 명령 큐 실행 스레드
# ============================================
command_queue = Queue()
is_executing = False

def exec_queue_loop():
    global is_executing
    while True:
        if not command_queue.empty():
            is_executing = True
            cmd = command_queue.get()
            try:
                kind = cmd.get("type")
                if kind == "move_joints":
                    move_joints(cmd.get("targets", {}), cmd.get("speed", 1.0), cmd.get("duration", 3.0))
                elif kind == "open_gripper":
                    open_gripper(cmd.get("speed", 0.5), cmd.get("duration", 2.0))
                elif kind == "close_gripper":
                    close_gripper(cmd.get("speed", 0.5), cmd.get("duration", 2.0))
                elif kind == "wait":
                    step_for(robot, timestep, float(cmd.get("seconds", 1.0)))
                else:
                    print(f"❓ Unknown command type: {kind}")
            except Exception as e:
                print("❌ Command exec error:", e)
            finally:
                command_queue.task_done()
                is_executing = False
        else:
            time.sleep(0.05)

threading.Thread(target=exec_queue_loop, daemon=True).start()
print("🚀 Command queue runner started")

# ============================================
# OpenAI
# ============================================
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"✅ OpenAI ready (model={OPENAI_MODEL})")
except Exception as e:
    client = None
    print("❌ OpenAI init failed:", e)

# ============================================
# 프리셋 포즈 (Fallback 매핑)
# ============================================
POSE_PRESETS = {
    "lift": {"shoulder_lift_joint": -1.0, "elbow_joint": 1.5},
    "down": {"shoulder_lift_joint": -0.6, "elbow_joint": 1.0},
    "home": {
        "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.57, "elbow_joint": 1.57,
        "wrist_1_joint": -1.57, "wrist_2_joint": 0.0, "wrist_3_joint": 0.0
    },
    "right": {"shoulder_pan_joint": 1.0},
    "left":  {"shoulder_pan_joint": -1.0},
}

def preset_from_utterance(text: str):
    t = (text or "").lower()
    if "원위치" in t or "홈" in t or "home" in t:
        return POSE_PRESETS["home"]
    if "들어올" in t or "lift" in t or "up" in t:
        return POSE_PRESETS["lift"]
    if "내려" in t or "down" in t:
        return POSE_PRESETS["down"]
    if "오른쪽" in t or "right" in t:
        return POSE_PRESETS["right"]
    if "왼쪽" in t or "left" in t:
        return POSE_PRESETS["left"]
    return None

# ============================================
# 계획 생성 (자연어 → JSON plan)
# ============================================
PLAN_SYSTEM = (
    "너는 UR10e 로봇팔 작업 계획자이자 실행 컨트롤러다.\n"
    "사용자의 한국어/영어 명령을 단계별 계획(JSON 배열)으로 반환하라.\n"
    "반드시 JSON 배열만 출력하고, 설명/텍스트를 추가하지 마라.\n"
    "각 단계는 다음 중 하나의 action을 가진다: "
    "'move_arm', 'control_gripper', 'wait'.\n"
    "각 단계는 반드시 'action'과 'params'를 포함한다.\n"
    "스키마 예시:\n"
    "[\n"
    "  {\"action\": \"move_arm\", \"params\": {\"targets\": {\"shoulder_lift_joint\": -1.0, \"elbow_joint\": 1.5}, \"speed\": 1.0, \"duration\": 2.5}},\n"
    "  {\"action\": \"control_gripper\", \"params\": {\"action\": \"close\", \"speed\": 0.5, \"duration\": 1.0}},\n"
    "  {\"action\": \"wait\", \"params\": {\"seconds\": 0.5}}\n"
    "]\n"
    "주의:\n"
    "- move_arm.targets는 dict 또는 [{'joint':..., 'angle':...}] 리스트 형태 모두 가능.\n"
    "- control_gripper.action은 'open' 또는 'close'.\n"
    "- 각 단계의 duration/seconds가 없으면 기본값을 생략 가능.\n"
)

def plan_from_text(user_message: str):
    """
    자연어 → JSON 계획 배열
    실패 시 빈 리스트 반환
    """
    # 간단한 의도에 대해선 로컬 프리셋으로 빠르게 반환
    preset = None
    if any(k in user_message for k in ["팔", "arm"]):
        preset = preset_from_utterance(user_message)
    if "그리퍼" in user_message or "gripper" in user_message:
        # 프리셋 + 그리퍼 결합 지시가 아니면 LLM 사용
        pass

    try:
        if client is None:
            # OpenAI 사용 불가 시, 프리셋만으로 대체
            if preset:
                return [{"action": "move_arm", "params": {"targets": preset}}]
            return []

        messages = [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": user_message},
        ]
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.2,
            max_completion_tokens=400,
        )
        content = resp.choices[0].message.content
        content = strip_code_fences(content)
        plan = json.loads(content)
        if isinstance(plan, list):
            return plan
        return []
    except Exception as e:
        print("⚠️ plan_from_text() failed, fallback:", e)
        if preset:
            return [{"action": "move_arm", "params": {"targets": preset}}]
        return []

# ============================================
# 계획 → 큐 등록
# ============================================
def enqueue_plan(plan: list):
    """
    계획(JSON 배열)을 큐 명령으로 변환하여 순차 실행되도록 등록
    허용 action: move_arm, control_gripper, wait
    """
    added = 0
    for step in plan:
        if not isinstance(step, dict):
            continue
        action = step.get("action")
        params = step.get("params", {})

        if action == "move_arm":
            raw_targets = params.get("targets")
            # list targets도 허용 (실행 시 변환)
            command_queue.put({
                "type": "move_joints",
                "targets": raw_targets if raw_targets else POSE_PRESETS["lift"],
                "speed": params.get("speed", 1.0),
                "duration": params.get("duration", 3.0),
            })
            added += 1

        elif action == "control_gripper":
            act = (params.get("action") or "").lower()
            typ = "open_gripper" if act == "open" else "close_gripper"
            command_queue.put({
                "type": typ,
                "speed": params.get("speed", 0.5),
                "duration": params.get("duration", 2.0),
            })
            added += 1

        elif action == "wait":
            secs = float(params.get("seconds", 1.0))
            command_queue.put({"type": "wait", "seconds": secs})
            added += 1

        else:
            print("❓ Unknown plan action:", action)

    return added

# ============================================
# 자연어 처리 엔드포인트 (WWI)
# ============================================
def handle_nl_command(text: str) -> str:
    """
    자연어 입력 → 계획 생성 → 큐 등록 → 결과 문자열
    """
    text = (text or "").strip()
    if not text:
        return "⚠️ 빈 명령입니다."

    # 특수: 아주 단순 지시(그리퍼 열/닫) 빠른 경로
    if text in ("그리퍼 열어", "그리퍼 열기", "open gripper"):
        command_queue.put({"type": "open_gripper"})
        return "✅ 즉시: 그리퍼 열기"
    if text in ("그리퍼 닫아", "그리퍼 닫기", "close gripper"):
        command_queue.put({"type": "close_gripper"})
        return "✅ 즉시: 그리퍼 닫기"

    # 계획 생성
    plan = plan_from_text(text)
    if not plan:
        # 마지막 보루: 프리셋 포즈
        preset = preset_from_utterance(text)
        if preset:
            command_queue.put({"type": "move_joints", "targets": preset})
            return f"✅ 프리셋 실행 등록: {preset}"
        return "⚠️ 계획 생성 실패: 이해할 수 없는 명령"

    added = enqueue_plan(plan)
    return f"✅ 계획 {len(plan)}단계 생성, 큐 등록 {added}개 완료"

# ============================================
# 메인 루프 (WWI)
# ============================================
print("🧠 Planner controller running (WWI enabled)")
while robot.step(timestep) != -1:
    msg = robot.wwiReceiveText()
    if msg:
        print(f"📩 USER: {msg}")
        result = handle_nl_command(msg)
        robot.wwiSendText(result)
