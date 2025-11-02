# controllers/ur10e_planner_controller/ur10e_planner_controller.py
from controller import Robot
from openai import OpenAI
import os
import dotenv
import json
import threading
import time
from queue import Queue
from datetime import datetime, timezone

# ============================================
# 설정
# ============================================
dotenv.load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LOG_PATH = os.getenv("PLAN_LOG_PATH", "ur10e_run_logs.jsonl")

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

def log_event(kind: str, data: dict):
    try:
        entry = {"t": datetime.now(timezone.utc).isoformat(), "kind": kind, **(data or {})}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass

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

    # 조인트 한계 보정
    def clamp(name: str, angle: float) -> float:
        m = motors.get(name)
        try:
            mn = m.getMinPosition(); mx = m.getMaxPosition()
            if mn is not None and mx is not None and mx >= mn:
                if angle < mn: return mn
                if angle > mx: return mx
        except Exception:
            pass
        return angle

    # 적용
    for name, angle in targets.items():
        m = motors.get(name)
        if not m:
            print(f"⚠️ Unknown joint '{name}', skip")
            continue
        try:
            m.setVelocity(abs(speed))
            a = clamp(name, float(angle))
            if a != angle:
                print(f"ℹ️ clamp {name}: {angle}→{a}")
            m.setPosition(a)
        except Exception as e:
            print(f"⚠️ setPosition failed for {name}: {e}")

    step_for(robot, timestep, duration)
    print(f"🦾 Joints moved → {targets}")
    log_event("exec_move_arm", {"targets": targets, "speed": speed, "duration": duration})

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
    log_event("exec_gripper", {"action": "open", "speed": speed, "duration": duration})

def close_gripper(speed=0.5, duration=2.0):
    for name in GRIPPER_NAMES:
        m = motors.get(name)
        if m: m.setVelocity(abs(speed))
    step_for(robot, timestep, duration)
    for name in GRIPPER_NAMES:
        m = motors.get(name)
        if m: m.setVelocity(0.0)
    print("✅ Gripper closed")
    log_event("exec_gripper", {"action": "close", "speed": speed, "duration": duration})

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
                    move_joints(cmd.get("targets", {}), cmd.get("speed", 1.0), cmd.get("duration", 0.5))
                elif kind == "open_gripper":
                    open_gripper(cmd.get("speed", 0.5), cmd.get("duration", 0.5))
                elif kind == "close_gripper":
                    close_gripper(cmd.get("speed", 0.5), cmd.get("duration", 0.5))
                elif kind == "wait":
                    step_for(robot, timestep, float(cmd.get("seconds", 1.0)))
                else:
                    print(f"❓ Unknown command type: {kind}")
                log_event("exec_step", {"cmd": cmd})
            except Exception as e:
                print("❌ Command exec error:", e)
                log_event("exec_error", {"cmd": cmd, "error": str(e)})
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

############################################
# 계획 생성 (Responses/Completions tools)
############################################
PLAN_SYSTEM = (
    "너는 UR10e 로봇팔 작업 계획자다. 반드시 함수 호출 produce_plan 을 사용하고,\n"
    "parameters.steps 배열 안에 단계들을 넣어라.\n"
    "각 단계는 {action, params}. action∈{move_arm, control_gripper, wait}.\n"
    "params.targets는 dict 또는 [{'joint','angle'}] 리스트 허용.\n"
    "control_gripper.params.action ∈ {'open','close'}."
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "produce_plan",
            "description": "사용자 명령을 실행 가능한 단계 배열로 변환",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {"type": "string", "enum": ["move_arm", "control_gripper", "wait"]},
                                "params": {
                                    "type": "object",
                                    "properties": {
                                        "targets": {
                                            "oneOf": [
                                                {
                                                    "type": "object",
                                                    "description": "조인트 이름별 각도 매핑 예시: {'shoulder_lift_joint': -1.0, 'elbow_joint': 1.5}"
                                                },
                                                {
                                                    "type": "array",
                                                    "description": "조인트 리스트 예시: [{'joint': 'shoulder_lift_joint', 'angle': -1.0}]",
                                                    "items": {
                                                        "type": "object",
                                                        "properties": {
                                                            "joint": {"type": "string"},
                                                            "angle": {"type": "number"}
                                                        },
                                                        "required": ["joint", "angle"]
                                                    }
                                                }
                                            ]
                                        },
                                        "speed": {"type": "number"},
                                        "duration": {"type": "number"},
                                        "seconds": {"type": "number"},
                                        "action": {"type": "string", "enum": ["open", "close"]}
                                    }
                                }
                            },
                            "required": ["action", "params"]
                        }
                    }
                },
                "required": ["steps"]
            }
        }
    }]

def plan_from_text(user_message: str):
    # 간단 의도 프리셋
    preset = preset_from_utterance(user_message)
    try:
        if client is None:
            if preset:
                return [{"action": "move_arm", "params": {"targets": preset}}]
            return []

        messages = [
            {"role": "system", "content": PLAN_SYSTEM},
            {"role": "user", "content": user_message},
        ]
        # Chat Completions + tools (required)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="required",
            temperature=0.2,
            max_completion_tokens=400,
        )
        msg = resp.choices[0].message
        tc = getattr(msg, "tool_calls", None)
        if tc and len(tc) > 0:
            args = tc[0].function.arguments
            try:
                obj = json.loads(strip_code_fences(args)) if isinstance(args, str) else args
                if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
                    return obj["steps"]
            except Exception as e:
                print("⚠️ tool args parse fail:", e)
        # 실패 시 프리셋
        if preset:
            return [{"action": "move_arm", "params": {"targets": preset}}]
        return []
    except Exception as e:
        print("⚠️ plan_from_text() tools failed:", e)
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
                "duration": params.get("duration", 0.5),
            })
            added += 1

        elif action == "control_gripper":
            act = (params.get("action") or "").lower()
            typ = "open_gripper" if act == "open" else "close_gripper"
            command_queue.put({
                "type": typ,
                "speed": params.get("speed", 0.5),
                "duration": params.get("duration", 0.5),
            })
            added += 1

        elif action == "wait":
            secs = float(params.get("seconds", 1.0))
            command_queue.put({"type": "wait", "seconds": secs})
            added += 1

        else:
            print("❓ Unknown plan action:", action)

    log_event("plan_enqueued", {"steps": plan, "added": added})
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
    if text in ("그리퍼 열어", "그리퍼 열기", "open gripper", "gripper open"):
        command_queue.put({"type": "open_gripper"})
        log_event("nl_shortcut", {"text": text, "action": "open_gripper"})
        return "✅ 즉시: 그리퍼 열기"
    if text in ("그리퍼 닫아", "그리퍼 닫기", "close gripper", "gripper close"):
        command_queue.put({"type": "close_gripper"})
        log_event("nl_shortcut", {"text": text, "action": "close_gripper"})
        return "✅ 즉시: 그리퍼 닫기"

    # 계획 생성
    plan = plan_from_text(text)
    log_event("plan_generated", {"text": text, "plan": plan})
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
        log_event("nl_received", {"text": msg})
        result = handle_nl_command(msg)
        robot.wwiSendText(result)
        log_event("nl_replied", {"reply": result})
