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
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "o3-mini")
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
            m.setPosition(float('inf'))  # velocity mode for gripper
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
    # list 형태도 수용 → dict로 정규화
    if isinstance(targets, list):
        targets = {
            normalize_joint_name(i["joint"]): i["angle"]
            for i in targets if isinstance(i, dict) and "joint" in i and "angle" in i
        }

    if not isinstance(targets, dict):
        print(f"⚠️ Invalid targets: {targets}")
        return

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
    log_event("exec_move", {"targets": targets, "speed": speed, "duration": duration})

def open_gripper(speed=1.0, duration=GRIPPER_DURATION):
    for n in GRIPPER_NAMES:
        m = motors.get(n)
        if m: m.setVelocity(-abs(speed))
    step_for(robot, timestep, duration)
    for n in GRIPPER_NAMES:
        m = motors.get(n)
        if m: m.setVelocity(0.0)
    log_event("exec_gripper", {"action": "open", "speed": speed, "duration": duration})

def close_gripper(speed=1.0, duration=GRIPPER_DURATION):
    for n in GRIPPER_NAMES:
        m = motors.get(n)
        if m: m.setVelocity(abs(speed))
    step_for(robot, timestep, duration)
    for n in GRIPPER_NAMES:
        m = motors.get(n)
        if m: m.setVelocity(0.0)
    log_event("exec_gripper", {"action": "close", "speed": speed, "duration": duration})

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
# 상태 인식 (ELLMER/EMOS-style)
# ============================================
def get_symbolic_state():
    """
    UR10e 현재 상태를 요약해 LLM에 전달할 수 있는 JSON+텍스트 형태로 반환
    - 관절 상태: getTargetPosition 값(목표값) 사용. 실제값 센서가 있다면 교체 권장.
    - 그리퍼 상태: 설정된 속도 기반 추정(간이).
    - EE 높이: 근사치(빠른 프롬프트용); 실제는 FK 사용 권장.
    """
    # 1) 관절 상태
    joint_state = {}
    for name in JOINT_NAMES:
        m = motors.get(name)
        try:
            # 목표값 기반(빠른 추정); PositionSensor 사용 시 실제값으로 대체 가능
            joint_state[name] = round(m.getTargetPosition(), 3)
        except Exception:
            joint_state[name] = None

    # 2) 그리퍼 상태(간이 추정)
    try:
        avg_vel = sum(abs(motors[g].getVelocity()) for g in GRIPPER_NAMES) / max(1, len(GRIPPER_NAMES))
        gripper_state = "closed_or_idle" if avg_vel < 0.01 else "moving/open"
    except Exception:
        gripper_state = "unknown"

    # 3) EE 높이 근사 (가벼운 휴리스틱)
    lift = joint_state.get("shoulder_lift_joint", 0.0) or 0.0
    ee_height = round(0.25 + (abs(lift) * 0.10), 3)  # 단순 근사치

    # 4) 프롬프트 친화 요약
    summary = (
        f"joints={joint_state}, gripper={gripper_state}, "
        f"end_effector_z≈{ee_height}m"
    )

    state = {
        "joints": joint_state,
        "gripper": gripper_state,
        "end_effector_z": ee_height,
        "summary": summary
    }
    log_event("state_snapshot", state)
    return state

# ============================================
# LLM 플랜 생성 (일반화 규칙, SafePlan 스타일)
# ============================================
PLAN_SYSTEM = """
너는 UR10e 산업용 로봇팔의 고수준 계획자다.
너의 역할은 주어진 목표(goal)와 현재 상태(state)를 바탕으로,
로봇이 안전하고 효율적으로 목표를 수행하도록 단계별 계획을 세우는 것이다.

[STATE VOCABULARY]
- shoulder_pan_joint: 수평 회전 축 (기본 방향 조정)
- shoulder_lift_joint: 어깨 상하 관절 (팔 높이 조절; 음수=아래, 양수=위)
- elbow_joint: 팔꿈치 관절 (팔 길이/접힘 조절; 작을수록 더 구부림)
- wrist_1_joint, wrist_2_joint, wrist_3_joint: 손목 3축 회전 (자세 정렬)
- gripper: open/closed_or_idle/moving/open (물체 잡기 여부; moving=open 동작 중일 수 있음)
- end_effector_z: 엔드이펙터의 높이 (m 단위; 0.0≈바닥)

[GENERAL PLANNING RULES]
1) 각 관절 각도는 [-3.14, 3.14] 범위를 넘지 않는다.
2) 엔드이펙터 높이는 0.05m 이상을 기본으로 유지해 충돌을 피한다.
3) 일반 순서: (a) 목표 위치/자세로 이동(move_arm) → (b) 필요시 그리퍼 동작(control_gripper) → (c) 완료 후 안정자세 복귀(선택).
4) 최소 단계와 짧은 경로로 계획한다. 불필요한 wait나 중복 이동을 넣지 않는다.
5) 이동 중 그리퍼는 닫지 않는다(안전 규칙).
6) 출력은 JSON이며, 각 단계는 {"action": "...", "params": {...}} 형식이다.
"""

# Responses API용 function tool 스키마
TOOLS = [
    {
        "type": "function",
        "name": "produce_plan",
        "description": "사용자 목표와 상태를 실행 가능한 단계 배열로 변환",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "enum": ["move_arm", "control_gripper", "wait"]
                            },
                            "params": {
                                "type": "object",
                                "properties": {
                                    "targets": {
                                        "oneOf": [
                                            {
                                                "type": "object",
                                                "description": "예: {'shoulder_lift_joint': -1.0, 'elbow_joint': 1.5}"
                                            },
                                            {
                                                "type": "array",
                                                "description": "예: [{'joint': 'shoulder_lift_joint','angle': -1.0}]",
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
                                    "action": {"type": "string", "enum": ["open", "close"]},
                                    "seconds": {"type": "number"},
                                    "speed": {"type": "number"},
                                    "duration": {"type": "number"}
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
]

def plan_from_text(msg: str):
    preset = preset_from_utterance(msg)
    # 오프라인(LMM 없음) 빠른 대응
    if client is None:
        plan = [{"action": "move_arm", "params": {"targets": preset}}] if preset else []
        print(f"🧩 Generated offline plan: {json.dumps(plan, ensure_ascii=False, indent=2)}")
        return plan

    try:
        # ✅ 상태 스냅샷 추가
        state = get_symbolic_state()

        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": PLAN_SYSTEM},
                {"role": "user", "content": f"현재 상태는 다음과 같다:\n{state['summary']}\n\n목표: {msg}\n\n이 목표를 수행하기 위한 단계별 계획을 JSON으로 작성해라."}
            ],
            tools=TOOLS,
            tool_choice="required",
            max_output_tokens=512,
        )

        plan = None
        for item in getattr(resp, "output", []):
            # reasoning 항목은 스킵
            if getattr(item, "type", None) == "reasoning":
                continue

            # function/tool call 파싱
            if getattr(item, "type", None) in ("function_call", "tool_call"):
                args_raw = getattr(item, "arguments", None)
                func_name = getattr(item, "name", None)
                if func_name != "produce_plan" or not args_raw:
                    continue

                print(f"🧩 Found tool_call '{func_name}'")
                try:
                    obj = json.loads(strip_code_fences(args_raw)) if isinstance(args_raw, str) else args_raw
                    if isinstance(obj, dict) and isinstance(obj.get("steps"), list):
                        plan = obj["steps"]
                        break
                except Exception as e:
                    print("⚠️ JSON parse error:", e)
                    continue

        if not plan and preset:
            plan = [{"action": "move_arm", "params": {"targets": preset}}]
            print("⚠️ Using preset fallback")

        # 출력 + 로그
        if plan:
            # list형 targets 정규화
            for s in plan:
                p = s.get("params", {})
                t = p.get("targets")
                if isinstance(t, list):
                    p["targets"] = {normalize_joint_name(i["joint"]): i["angle"] for i in t if isinstance(i, dict) and "joint" in i and "angle" in i}
                s["params"] = p

            print("🧠 LLM Generated Plan:")
            for i, step in enumerate(plan, 1):
                print(f"  {i}. {step.get('action')} | {step.get('params')}")
            log_event("plan_generated", {"input": msg, "plan": plan, "state": state})
            return plan
        else:
            print("⚠️ No plan found in response, returning empty plan")

    except Exception as e:
        print("❌ plan_from_text() exception:", e)

    # 완전 fallback
    if preset:
        return [{"action": "move_arm", "params": {"targets": preset}}]
    return []

# ============================================
# 큐 등록
# ============================================
def enqueue_plan(plan):
    if not plan:
        return
    for s in plan:
        a, p = s.get("action"), s.get("params",{})
        if a == "move_arm":
            targets = p.get("targets", POSE_PRESETS["lift"])
            if isinstance(targets, list):
                targets = {normalize_joint_name(i["joint"]): i["angle"] for i in targets if isinstance(i, dict) and "joint" in i and "angle" in i}
            command_queue.put({
                "type":"move_joints",
                "targets": targets,
                "speed": float(p.get("speed", 2.0)),
                "duration": float(p.get("duration", MOVE_DURATION))
            })
        elif a == "control_gripper":
            act = (p.get("action","").lower())
            command_queue.put({"type": "open_gripper" if act == "open" else "close_gripper"})
        elif a == "wait":
            command_queue.put({"type":"wait","seconds":min(0.1, float(p.get("seconds",0.1)))})

# ============================================
# 메인 루프 (WWI)
# ============================================
print("🟢 Ready for WWI commands")
while robot.step(timestep) != -1:
    msg = robot.wwiReceiveText()
    if not msg:
        continue
    print(f"📩 USER: {msg}")
    plan = plan_from_text(msg)
    enqueue_plan(plan)
    robot.wwiSendText(f"✅ {len(plan)}단계수행 중")
