# controllers/vlm_controller/vlm_controller.py
# CrewAI 기반 멀티 에이전트 가사 로봇 컨트롤러
from controller import Robot
from openai import OpenAI
import os, dotenv, json, threading, time
from queue import Queue, Empty
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional

# CrewAI imports
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Type

# ============================================
# 설정
# ============================================
dotenv.load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
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
sensors = {}  # PositionSensor 추가

# 초기 홈 포지션 (NaN 방지)
HOME_POSITION = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.57,
    "elbow_joint": 1.57,
    "wrist_1_joint": -1.57,
    "wrist_2_joint": 0.0,
    "wrist_3_joint": 0.0
}

for n in JOINT_NAMES + GRIPPER_NAMES:
    try:
        m = robot.getDevice(n)
        if n in GRIPPER_NAMES:
            m.setPosition(float('inf'))  # velocity mode for gripper
            m.setVelocity(0.0)
        else:
            # 초기 위치 설정 (NaN 방지)
            initial_pos = HOME_POSITION.get(n, 0.0)
            m.setPosition(initial_pos)
            m.setVelocity(1.0)
        motors[n] = m
        
        # PositionSensor 활성화 (실제 관절 각도 읽기용)
        if n in JOINT_NAMES:
            try:
                sensor_name = f"{n}_sensor"  # 일반적인 naming convention
                sensor = robot.getDevice(sensor_name)
                if sensor:
                    sensor.enable(timestep)
                    sensors[n] = sensor
                    print(f"✅ Sensor enabled: {sensor_name}")
                else:
                    print(f"⚠️ No sensor found for {n}, using target position")
                    sensors[n] = None
            except Exception as e:
                # 센서가 없으면 목표값 사용
                print(f"⚠️ Sensor error for {n}: {e}")
                sensors[n] = None
    except Exception as e:
        print(f"[WARN] Device init failed: {n} ({e})")

print("✅ Motors:", list(motors.keys()))
print("✅ Sensors:", [k for k, v in sensors.items() if v is not None])

# 초기 안정화를 위한 몇 step 실행
print("⏳ 초기화 중... 센서 안정화 대기")
for _ in range(10):
    if robot.step(timestep) == -1:
        break
print("✅ 초기화 완료")

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
# 포즈 프리셋
# ============================================
POSE_PRESETS = {
    "lift": {"shoulder_lift_joint": -1.0, "elbow_joint": 1.5},
    "down": {"shoulder_lift_joint": -0.6, "elbow_joint": 1.0},
    "home": HOME_POSITION.copy(),  # 홈 포지션 재사용
    "reach_table": {"shoulder_pan_joint": 0.5, "shoulder_lift_joint": -0.8, "elbow_joint": 1.2},
    "reach_shelf": {"shoulder_pan_joint": 0.0, "shoulder_lift_joint": -1.8, "elbow_joint": 2.0},
}

# ============================================
# 상태 인식
# ============================================
def get_symbolic_state() -> Dict[str, Any]:
    """
    UR10e 현재 상태를 요약해 LLM에 전달할 수 있는 JSON+텍스트 형태로 반환
    """
    # 1) 관절 상태 (실제 센서값 우선, 없으면 목표값)
    joint_state = {}
    for name in JOINT_NAMES:
        m = motors.get(name)
        s = sensors.get(name)
        value = None
        
        try:
            if s is not None:
                # PositionSensor가 있으면 실제 각도 사용
                value = s.getValue()
            elif m is not None:
                # 센서가 없으면 목표값 사용 (fallback)
                value = m.getTargetPosition()
            
            # NaN 체크
            if value is not None and not (value != value):  # NaN check
                joint_state[name] = round(value, 3)
            else:
                # NaN이면 홈 포지션 사용
                joint_state[name] = HOME_POSITION.get(name, 0.0)
        except Exception as e:
            # 에러 발생 시 홈 포지션 사용
            joint_state[name] = HOME_POSITION.get(name, 0.0)

    # 2) 그리퍼 상태(간이 추정)
    try:
        avg_vel = sum(abs(motors[g].getVelocity()) for g in GRIPPER_NAMES) / max(1, len(GRIPPER_NAMES))
        gripper_state = "closed" if avg_vel < 0.01 else "open"
    except Exception:
        gripper_state = "unknown"

    # 3) EE 높이 근사
    lift = joint_state.get("shoulder_lift_joint", 0.0)
    if lift is None or (lift != lift):  # None or NaN check
        lift = 0.0
    ee_height = round(0.25 + (abs(lift) * 0.10), 3)

    # 4) 프롬프트 친화 요약
    summary = (
        f"관절 상태: {joint_state}\n"
        f"그리퍼: {gripper_state}\n"
        f"엔드이펙터 높이: {ee_height}m"
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
# CrewAI 커스텀 도구 (BaseTool 상속)
# ============================================

# 1. 로봇 상태 조회 도구
class GetRobotStateInput(BaseModel):
    """Input schema for GetRobotStateTool."""
    query: str = Field(default="", description="Optional query parameter (not used)")

class GetRobotStateTool(BaseTool):
    name: str = "get_robot_state"
    description: str = "현재 로봇의 상태(관절 각도, 그리퍼 상태, 엔드이펙터 높이)를 JSON 형식으로 가져옵니다."
    args_schema: Type[BaseModel] = GetRobotStateInput

    def _run(self, query: str = "") -> str:
        state = get_symbolic_state()
        return json.dumps(state, ensure_ascii=False, indent=2)


# 2. 관절 각도 검증 도구
class ValidateJointAnglesInput(BaseModel):
    """Input schema for ValidateJointAnglesTool."""
    angles: str = Field(..., description="JSON 문자열 형식의 관절 각도 딕셔너리. 예: '{\"shoulder_lift_joint\": -1.0}'")

class ValidateJointAnglesTool(BaseTool):
    name: str = "validate_joint_angles"
    description: str = "관절 각도가 안전 범위 [-3.14, 3.14] 내에 있는지 검증합니다."
    args_schema: Type[BaseModel] = ValidateJointAnglesInput

    def _run(self, angles: str) -> str:
        try:
            angles_dict = json.loads(angles)
            issues = []
            
            for joint, angle in angles_dict.items():
                if not isinstance(angle, (int, float)):
                    issues.append(f"{joint}: 각도는 숫자여야 합니다")
                    continue
                if angle < -3.14 or angle > 3.14:
                    issues.append(f"{joint}: {angle}는 [-3.14, 3.14] 범위를 벗어남")
            
            if issues:
                return f"❌ 검증 실패:\n" + "\n".join(issues)
            return "✅ 모든 관절 각도가 안전 범위 내에 있습니다."
        except Exception as e:
            return f"❌ 검증 오류: {e}"


# 3. 계획 안전성 검증 도구
class ValidatePlanSafetyInput(BaseModel):
    """Input schema for ValidatePlanSafetyTool."""
    plan: str = Field(..., description="JSON 문자열 형식의 실행 계획. 예: '[{\"action\": \"move_arm\", \"params\": {...}}]'")

class ValidatePlanSafetyTool(BaseTool):
    name: str = "validate_plan_safety"
    description: str = "전체 실행 계획의 안전성을 검증합니다. 충돌, 각도 범위, 동작 순서 등을 확인합니다."
    args_schema: Type[BaseModel] = ValidatePlanSafetyInput

    def _run(self, plan: str) -> str:
        try:
            plan_list = json.loads(plan)
            issues = []
            
            # 규칙 1: 이동 중 그리퍼 닫기 금지
            for i, step in enumerate(plan_list):
                action = step.get("action")
                if i > 0:
                    prev_action = plan_list[i-1].get("action")
                    if prev_action == "move_arm" and action == "control_gripper":
                        params = step.get("params", {})
                        if params.get("action") == "close":
                            issues.append(f"단계 {i+1}: 이동 직후 그리퍼를 닫으면 안전하지 않습니다")
            
            # 규칙 2: 최소 단계 수 확인
            if len(plan_list) == 0:
                issues.append("계획이 비어 있습니다")
            
            # 규칙 3: 관절 각도 범위 검증
            for i, step in enumerate(plan_list):
                if step.get("action") == "move_arm":
                    targets = step.get("params", {}).get("targets", {})
                    for joint, angle in targets.items():
                        if isinstance(angle, (int, float)) and (angle < -3.14 or angle > 3.14):
                            issues.append(f"단계 {i+1}: {joint} 각도 {angle}가 범위 벗어남")
            
            if issues:
                return f"❌ 안전성 검증 실패:\n" + "\n".join(issues)
            return f"✅ 계획이 안전합니다. 총 {len(plan_list)}단계"
        except Exception as e:
            return f"❌ 검증 오류: {e}"


# 4. 프리셋 포즈 조회 도구
class GetPresetPosesInput(BaseModel):
    """Input schema for GetPresetPosesTool."""
    query: str = Field(default="", description="Optional query parameter (not used)")

class GetPresetPosesTool(BaseTool):
    name: str = "get_preset_poses"
    description: str = "사용 가능한 미리 정의된 로봇 포즈 목록(home, lift, down, reach_table, reach_shelf)을 가져옵니다."
    args_schema: Type[BaseModel] = GetPresetPosesInput

    def _run(self, query: str = "") -> str:
        return json.dumps(POSE_PRESETS, ensure_ascii=False, indent=2)


# 도구 인스턴스 생성
get_robot_state_tool = GetRobotStateTool()
validate_joint_angles_tool = ValidateJointAnglesTool()
validate_plan_safety_tool = ValidatePlanSafetyTool()
get_preset_poses_tool = GetPresetPosesTool()

# ============================================
# CrewAI 에이전트 정의
# ============================================
try:
    llm = ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        temperature=0.2
    )
    print(f"✅ LLM ready ({OPENAI_MODEL})")
except Exception as e:
    llm = None
    print("❌ LLM init failed:", e)

def create_agents():
    """멀티 에이전트 생성: Planner, Executor, Validator"""
    
    # 1) Planner Agent - 고수준 작업 계획 수립
    planner = Agent(
        role='가사 로봇 작업 계획자 (Task Planner)',
        goal='사용자 요청을 분석하고 안전하고 효율적인 로봇 작업 계획을 수립합니다.',
        backstory="""당신은 가정용 로봇 시스템의 두뇌입니다. 
        사용자의 일상적인 요청(물건 가져오기, 정리하기 등)을 이해하고, 
        로봇이 수행할 단계별 작업을 계획합니다. 
        안전성과 효율성을 최우선으로 고려합니다.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
        tools=[get_robot_state_tool, get_preset_poses_tool]
    )
    
    # 2) Executor Agent - 계획을 실행 가능한 명령으로 변환
    executor = Agent(
        role='로봇 제어 실행자 (Robot Controller)',
        goal='계획을 UR10e 로봇이 실행 가능한 구체적인 명령으로 변환합니다.',
        backstory="""당신은 UR10e 산업용 로봇팔의 전문가입니다.
        6개 관절(shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)과 
        3핑거 그리퍼를 제어합니다. 
        추상적인 계획을 정밀한 관절 각도와 그리퍼 명령으로 변환하는 것이 전문입니다.
        
        [실행 명령 형식]
        - move_arm: 관절을 목표 각도로 이동
        - control_gripper: 그리퍼 열기/닫기
        - wait: 대기 시간
        
        모든 관절 각도는 [-3.14, 3.14] 범위 내여야 합니다.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[get_robot_state_tool, validate_joint_angles_tool, get_preset_poses_tool]
    )
    
    # 3) Validator Agent - 실행 전 안전성 검증
    validator = Agent(
        role='안전성 검증자 (Safety Validator)',
        goal='로봇 작업 계획과 명령의 안전성을 검증하고 승인합니다.',
        backstory="""당신은 로봇 안전 시스템의 마지막 방어선입니다.
        모든 명령이 실행되기 전에 안전성을 검증합니다.
        
        [검증 규칙]
        1. 관절 각도가 [-3.14, 3.14] 범위 내인가?
        2. 엔드이펙터가 충돌 위험 영역에 있지 않은가?
        3. 이동 중 그리퍼를 닫지 않는가?
        4. 계획이 논리적으로 타당한가?
        
        문제가 발견되면 즉시 거부하고 수정을 요청합니다.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        tools=[validate_joint_angles_tool, validate_plan_safety_tool, get_robot_state_tool]
    )
    
    return planner, executor, validator

# ============================================
# CrewAI Crew 실행
# ============================================
def execute_with_crew(user_goal: str) -> Optional[List[Dict]]:
    """
    CrewAI 멀티 에이전트 시스템으로 작업 실행
    
    Args:
        user_goal: 사용자 목표 (예: "컵을 집어서 테이블에 놓아줘")
    
    Returns:
        실행 계획 리스트 또는 None
    """
    if llm is None:
        print("❌ LLM이 초기화되지 않았습니다")
        return None
    
    try:
        # 에이전트 생성
        planner, executor, validator = create_agents()
        
        # 현재 상태 가져오기
        current_state = get_symbolic_state()
        
        # Task 1: 계획 수립
        planning_task = Task(
            description=f"""
            사용자 목표: "{user_goal}"
            
            현재 로봇 상태:
            {current_state['summary']}
            
            이 목표를 달성하기 위한 고수준 작업 계획을 수립하세요.
            각 단계를 명확히 정의하고, 안전성을 고려하세요.
            """,
            agent=planner,
            expected_output="단계별 작업 계획 (텍스트 형식)"
        )
        
        # Task 2: 실행 명령 생성
        execution_task = Task(
            description="""
            Planner가 수립한 계획을 UR10e 로봇이 실행 가능한 명령으로 변환하세요.
            
            출력 형식은 반드시 다음 JSON 배열이어야 합니다:
            [
                {
                    "action": "move_arm",
                            "params": {
                        "targets": {"shoulder_lift_joint": -1.0, "elbow_joint": 1.5},
                        "speed": 2.0,
                        "duration": 0.3
                    }
                },
                {
                    "action": "control_gripper",
                    "params": {"action": "open"}
                                            }
                                        ]
            
            가능한 action: move_arm, control_gripper, wait
            모든 관절 각도는 [-3.14, 3.14] 범위 내여야 합니다.
            """,
            agent=executor,
            expected_output="JSON 형식의 실행 명령 배열",
            context=[planning_task]
        )
        
        # Task 3: 안전성 검증
        validation_task = Task(
            description="""
            Executor가 생성한 실행 명령의 안전성을 검증하세요.
            
            검증 항목:
            1. 모든 관절 각도가 안전 범위 내인가?
            2. 충돌 위험이 없는가?
            3. 이동 중 그리퍼 닫기 같은 위험한 동작이 없는가?
            4. 계획이 논리적으로 타당한가?
            
            검증 통과 시: "APPROVED: " + 원본 JSON 명령을 그대로 출력
            검증 실패 시: "REJECTED: " + 거부 사유
            """,
            agent=validator,
            expected_output="검증 결과 및 승인된 실행 계획",
            context=[execution_task]
        )
        
        # Crew 생성 및 실행
        crew = Crew(
            agents=[planner, executor, validator],
            tasks=[planning_task, execution_task, validation_task],
            process=Process.sequential,  # 순차 실행
            verbose=True
        )
        
        print("\n🚀 CrewAI 멀티 에이전트 시스템 시작...")
        print("=" * 60)
        
        result = crew.kickoff()
        
        print("=" * 60)
        print(f"🎯 Crew 실행 결과:\n{result}")
        
        # 결과 파싱
        result_str = str(result)
        
        # APPROVED 체크
        if "APPROVED:" in result_str:
            # JSON 추출
            json_start = result_str.find("[")
            json_end = result_str.rfind("]") + 1
            if json_start != -1 and json_end > json_start:
                json_str = result_str[json_start:json_end]
                plan = json.loads(json_str)
                
                print(f"\n✅ 검증 통과! {len(plan)}단계 계획:")
            for i, step in enumerate(plan, 1):
                print(f"  {i}. {step.get('action')} | {step.get('params')}")
                
                log_event("crew_approved", {"goal": user_goal, "plan": plan})
                return plan
        
        if "REJECTED:" in result_str:
            print(f"\n❌ 검증 거부됨")
            log_event("crew_rejected", {"goal": user_goal, "reason": result_str})
            return None
        
        # Fallback: JSON 직접 파싱 시도
        json_start = result_str.find("[")
        json_end = result_str.rfind("]") + 1
        if json_start != -1 and json_end > json_start:
            json_str = result_str[json_start:json_end]
            plan = json.loads(json_str)
            print(f"\n⚠️ Fallback 파싱 성공: {len(plan)}단계")
            return plan
        
        print("\n⚠️ 유효한 계획을 추출할 수 없습니다")
        return None

    except Exception as e:
        print(f"❌ CrewAI 실행 오류: {e}")
        import traceback
        traceback.print_exc()
        return None

# ============================================
# 큐 등록
# ============================================
def enqueue_plan(plan: List[Dict]):
    """실행 계획을 명령 큐에 등록"""
    if not plan:
        return
    
    for s in plan:
        a, p = s.get("action"), s.get("params", {})
        if a == "move_arm":
            targets = p.get("targets", POSE_PRESETS["home"])
            if isinstance(targets, list):
                targets = {
                    normalize_joint_name(i["joint"]): i["angle"] 
                    for i in targets 
                    if isinstance(i, dict) and "joint" in i and "angle" in i
                }
            command_queue.put({
                "type": "move_joints",
                "targets": targets,
                "speed": float(p.get("speed", 2.0)),
                "duration": float(p.get("duration", MOVE_DURATION))
            })
        elif a == "control_gripper":
            act = (p.get("action", "").lower())
            command_queue.put({
                "type": "open_gripper" if act == "open" else "close_gripper"
            })
        elif a == "wait":
            command_queue.put({
                "type": "wait",
                "seconds": min(0.1, float(p.get("seconds", 0.1)))
            })

# ============================================
# 메인 루프 (WWI)
# ============================================
print("🟢 CrewAI 멀티 에이전트 가사 로봇 시스템 준비 완료")
print("   Agents: Planner → Executor → Validator")

msg = '캔 집어서 옮겨줘'

time.sleep(5)

print(f"\n📩 사용자 요청: {msg}")

# CrewAI 멀티 에이전트 실행
plan = execute_with_crew(msg)
    enqueue_plan(plan)
