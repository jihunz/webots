# controllers/ur10e_planner_controller/ur10e_planner_controller.py
from controller import Robot
from openai import OpenAI
import os, dotenv, json, threading, time
from queue import Queue, Empty
from datetime import datetime, timezone

# ============================================
# 설정
# ============================================
dotenv.load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LOG_PATH = os.getenv("PLAN_LOG_PATH", "ur10e_run_logs.jsonl")

# 초고속 설정
MOVE_DURATION = 0.3
GRIPPER_DURATION = 0.25
QUEUE_TIMEOUT = 0.001
MIN_STEPS = 3

# ============================================
# 공통 유틸
# ============================================
def strip_code_fences(s: str):
    if not s:
        return s
    s = s.strip()
    if s.startswith("```"):
        s = s.split("```", 2)
        if len(s) == 3:
            return s[1].split("\n", 1)[-1] if s[1].startswith(("json", "JSON")) else s[1]
    return s

def step_for(robot: Robot, timestep: int, duration: float, min_steps: int = MIN_STEPS):
    """최소한의 step만 돌고 빠르게 다음 단계로 넘어감"""
    end = time.time() + max(0.0, duration)
    steps = 0
    while time.time() < end or steps < min_steps:
        if robot.step(timestep) == -1:
            break
        steps += 1

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
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
]
GRIPPER_NAMES = [
    "finger_1_joint_1", "finger_2_joint_1", "finger_middle_joint_1"
]

motors = {}
for n in JOINT_NAMES + GRIPPER_NAMES:
    try:
        m = robot.getDevice(n)
        if n in GRIPPER_NAMES:
            m.setPosition(float('inf'))
            m.setVelocity(0.0)
        else:
            m.setVelocity(1.0)
        motors[n] = m
    except Exception as e:
        print(f"[WARN] Device init failed: {n} ({e})")

print("✅ Motors:", list(motors.keys()))

# ============================================
# 이름 매핑 (LLM → 실제 UR10e)
# ============================================
JOINT_ALIAS = {
    "base": "shoulder_pan_joint",
    "shoulder": "shoulder_lift_joint",
    "elbow": "elbow_joint",
    "wrist": "wrist_1_joint",
    "wrist_1": "wrist_1_joint",
    "wrist_2": "wrist_2_joint",
    "wrist_3": "wrist_3_joint",
    "pan": "shoulder_pan_joint",
    "lift": "shoulder_lift_joint",
    "roll": "wrist_3_joint",
}

def normalize_joint_name(name: str) -> str:
    n = (name or "").lower().strip()
    return JOINT_ALIAS.get(n, name)

# ============================================
# 제어 함수
# ============================================
def move_joints(targets, speed=2.0, duration=MOVE_DURATION):
    if isinstance(targets, list):
        targets = {
            normalize_joint_name(i["joint"]): i["angle"]
            for i in targets if "joint" in i and "angle" in i
        }

    for n, a in targets.items():
        real_name = normalize_joint_name(n)
        m = motors.get(real_name)
        if not m:
            print(f"⚠️ Unknown joint: {n} (→ {real_name})")
            continue
        try:
            m.setVelocity(abs(speed))
            m.setPosition(float(a))
        except Exception as e:
            print(f"⚠️ setPosition fail: {n} ({e})")

    step_for(robot, timestep, duration)
    log_event("exec_move", {"targets": targets})

def open_gripper(speed=1.0, duration=GRIPPER_DURATION):
    for n in GRIPPER_NAMES:
        m = motors.get(n)
        if m: m.setVelocity(-abs(speed))
    step_for(robot, timestep, duration)
    for n in GRIPPER_NAMES:
        m = motors.get(n)
        if m: m.setVelocity(0.0)
    log_event("exec_gripper", {"action": "open"})

def close_gripper(speed=1.0, duration=GRIPPER_DURATION):
    for n in GRIPPER_NAMES:
        m = motors.get(n)
        if m: m.setVelocity(abs(speed))
    step_for(robot, timestep, duration)
    for n in GRIPPER_NAMES:
        m = motors.get(n)
        if m: m.setVelocity(0.0)
    log_event("exec_gripper", {"action": "close"})

# ============================================
# 명령 큐
# ============================================
command_queue = Queue()

def exec_queue_loop():
    while True:
        try:
            cmd = command_queue.get(timeout=QUEUE_TIMEOUT)
        except Empty:
            continue
        t = cmd.get("type")
        try:
            if t == "move_joints":
                move_joints(cmd["targets"], cmd.get("speed", 2.0), cmd.get("duration", MOVE_DURATION))
            elif t == "open_gripper": open_gripper()
            elif t == "close_gripper": close_gripper()
            elif t == "wait": step_for(robot, timestep, cmd.get("seconds", 0.1))
        except Exception as e:
            print("❌ Exec error:", e)
        finally:
            command_queue.task_done()

threading.Thread(target=exec_queue_loop, daemon=True).start()
print("🚀 Queue runner (ultra-fast mode) started")

# ============================================
# OpenAI 초기화
# ============================================
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print(f"✅ OpenAI ready ({OPENAI_MODEL})")
except Exception as e:
    client = None
    print("❌ OpenAI init failed:", e)

# ============================================
# 포즈 프리셋
# ============================================
POSE_PRESETS = {
    "lift": {"shoulder_lift_joint": -1.0, "elbow_joint": 1.5},
    "down": {"shoulder_lift_joint": -0.6, "elbow_joint": 1.0},
    "home": {
        "shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.57,
        "elbow_joint": 1.57, "wrist_1_joint": -1.57,
        "wrist_2_joint": 0.0, "wrist_3_joint": 0.0
    },
}

def preset_from_utterance(t: str):
    t = (t or "").lower()
    if "홈" in t or "home" in t: return POSE_PRESETS["home"]
    if "들어올" in t or "lift" in t or "up" in t: return POSE_PRESETS["lift"]
    if "내려" in t or "down" in t: return POSE_PRESETS["down"]
    return None

# ============================================
# LLM 플랜 생성
# ============================================
PLAN_SYSTEM = (
    "너는 UR10e 로봇팔 계획자다. 반드시 아래 조인트 이름만 사용해야 한다:\n"
    "['shoulder_pan_joint','shoulder_lift_joint','elbow_joint','wrist_1_joint','wrist_2_joint','wrist_3_joint'].\n"
    "불필요한 wait 단계는 포함하지 말고 가능한 한 빠르게 수행하라.\n"
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "produce_plan",
        "description": "사용자 명령을 실행 가능한 단계로 변환",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "enum": ["move_arm","control_gripper","wait"]},
                            "params": {
                                "type": "object",
                                "properties": {
                                    "targets": {
                                        "oneOf": [
                                            {"type": "object"},
                                            {"type": "array", "items": {
                                                "type": "object",
                                                "properties": {
                                                    "joint": {"type": "string"},
                                                    "angle": {"type": "number"}
                                                },
                                                "required": ["joint", "angle"]
                                            }}
                                        ]
                                    },
                                    "action": {"type": "string", "enum": ["open","close"]},
                                    "seconds": {"type": "number"}
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

def plan_from_text(msg: str):
    preset = preset_from_utterance(msg)
    if client is None:
        plan = [{"action": "move_arm", "params": {"targets": preset}}] if preset else []
        print(f"🧩 Generated offline plan: {json.dumps(plan, ensure_ascii=False, indent=2)}")
        return plan
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": msg}],
            tools=TOOLS, tool_choice="required", temperature=0.1, max_completion_tokens=400,
        )
        tc = resp.choices[0].message.tool_calls
        if tc:
            args = json.loads(strip_code_fences(tc[0].function.arguments))
            plan = args.get("steps", [])
            # ✅ 계획 시각화 출력
            print("🧠 LLM Generated Plan:")
            for i, step in enumerate(plan, start=1):
                print(f"  {i}. action={step.get('action')} | params={step.get('params')}")
            log_event("plan_generated", {"input": msg, "plan": plan})
            return plan
    except Exception as e:
        print("⚠️ plan_from_text:", e)
    plan = [{"action": "move_arm", "params": {"targets": preset}}] if preset else []
    print(f"🧩 Fallback plan: {json.dumps(plan, ensure_ascii=False, indent=2)}")
    return plan

# ============================================
# 큐 등록
# ============================================
def enqueue_plan(plan):
    for s in plan:
        a, p = s.get("action"), s.get("params",{})
        if a=="move_arm":
            command_queue.put({"type":"move_joints","targets":p.get("targets",POSE_PRESETS["lift"]),
                               "speed":2.0,"duration":MOVE_DURATION})
        elif a=="control_gripper":
            act=p.get("action","").lower()
            command_queue.put({"type":"open_gripper" if act=="open" else "close_gripper"})
        elif a=="wait":
            command_queue.put({"type":"wait","seconds":min(0.1,p.get("seconds",0.1))})

# ============================================
# 메인 루프 (WWI)
# ============================================
print("🧠 Ultra-fast planner running")
while robot.step(timestep) != -1:
    msg = robot.wwiReceiveText()
    if not msg: continue
    print(f"📩 USER: {msg}")
    plan = plan_from_text(msg)
    enqueue_plan(plan)
    robot.wwiSendText(f"✅ {len(plan)}단계 초고속 수행 중")
