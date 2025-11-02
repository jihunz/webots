from controller import Robot
from openai import OpenAI
import os
import time
import json
import dotenv


# ---------------- Device name constants (from IprHd6ms180.proto) ----------------

JOINT_MOTOR_NAMES = [
    "base",
    "upperarm",
    "forearm",
    "wrist",
    "rotational_wrist",
]

GRIPPER_MOTOR_NAMES = [
    "gripper::left",
    "gripper::right",
]

JOINT_SENSOR_NAMES = [
    "base_sensor",
    "upperarm_sensor",
    "forearm_sensor",
    "wrist_sensor",
    "rotational_wrist_sensor",
    "gripper::left_sensor",
    "gripper::right_sensor",
]

IR_SENSOR_NAMES = [
    "ds0", "ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds7", "ds8"
]

TOUCH_SENSOR_NAMES = [
    "ts0", "ts1", "ts2", "ts3"
]

HOME_POSE = {
    "base": 0.0,
    "upperarm": 0.0,
    "forearm": 0.0,
    "wrist": 0.0,
    "rotational_wrist": 0.0,
}


# ---------------- Helpers ----------------

def clamp_to_limits(motor, target):
    try:
        min_pos = motor.getMinPosition()
        max_pos = motor.getMaxPosition()
        if min_pos is None or max_pos is None:
            return target
        if max_pos < min_pos:
            return target
        if target < min_pos:
            return min_pos
        if target > max_pos:
            return max_pos
        return target
    except Exception:
        return target


def wait_until_targets_reached(robot, timestep_ms, sensors, targets_by_name, tolerance=0.01, timeout_s=5.0):
    start = time.time()
    while time.time() - start < timeout_s:
        all_ok = True
        for name, target in targets_by_name.items():
            sensor_name = f"{name}_sensor"
            sensor = sensors.get(sensor_name)
            if sensor is None:
                continue
            current = sensor.getValue()
            if abs(current - target) > tolerance:
                all_ok = False
                break
        if all_ok:
            return True
        if robot.step(timestep_ms) == -1:
            break
    return False


# ---------------- LLM Tool (Function) Schemas ----------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_joints",
            "description": "여러 관절을 라디안 단위 목표각으로 이동. 도달 대기 옵션 지원.",
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "관절명→라디안: base, upperarm, forearm, wrist, rotational_wrist"
                    },
                    "speed": {"type": "number"},
                    "wait": {"type": "boolean", "default": True},
                    "tolerance": {"type": "number", "default": 0.01},
                    "timeout": {"type": "number", "default": 5}
                },
                "required": ["targets"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_joint_delta",
            "description": "단일 관절을 현재값 기준 라디안 증분만큼 이동.",
            "parameters": {
                "type": "object",
                "properties": {
                    "joint": {"type": "string", "enum": JOINT_MOTOR_NAMES},
                    "delta": {"type": "number"},
                    "speed": {"type": "number"},
                    "wait": {"type": "boolean", "default": True},
                    "tolerance": {"type": "number", "default": 0.01},
                    "timeout": {"type": "number", "default": 5}
                },
                "required": ["joint", "delta"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_joint_velocity_limits",
            "description": "관절별 속도상한(rad/s) 설정.",
            "parameters": {"type": "object", "properties": {"limits": {"type": "object", "additionalProperties": {"type": "number"}}}, "required": ["limits"]}
        }
    },
    {"type": "function", "function": {"name": "open_gripper", "description": "그리퍼를 최대 개도로 염.", "parameters": {"type": "object", "properties": {"speed": {"type": "number"}, "wait": {"type": "boolean", "default": True}, "timeout": {"type": "number", "default": 3}}}}},
    {"type": "function", "function": {"name": "close_gripper", "description": "그리퍼를 닫음. 접촉 감지 시 정지 옵션.", "parameters": {"type": "object", "properties": {"speed": {"type": "number"}, "stop_on_contact": {"type": "boolean", "default": True}, "wait": {"type": "boolean", "default": True}, "timeout": {"type": "number", "default": 3}}}}},
    {"type": "function", "function": {"name": "set_gripper", "description": "그리퍼 좌/우 각도(rad) 설정. 'both'로 양손 동일 설정.", "parameters": {"type": "object", "properties": {"left": {"type": "number"}, "right": {"type": "number"}, "both": {"type": "number"}, "speed": {"type": "number"}, "wait": {"type": "boolean", "default": True}, "timeout": {"type": "number", "default": 3}}}}},
    {"type": "function", "function": {"name": "read_joint_positions", "description": "모든 관절/그리퍼 각도(rad) 읽기.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_gripper_contacts", "description": "터치센서 ts0..ts3 값을 읽기.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_proximity", "description": "IR 근접센서 ds0..ds8 값을 읽기.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "wait_until_reached", "description": "지정 관절이 목표 각도 도달할 때까지 대기.", "parameters": {"type": "object", "properties": {"targets": {"type": "object", "additionalProperties": {"type": "number"}}, "tolerance": {"type": "number", "default": 0.01}, "timeout": {"type": "number", "default": 5}}, "required": ["targets"]}}},
    {"type": "function", "function": {"name": "stop_all_motors", "description": "모든 모터 정지 및 현재 위치 유지.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "go_home_pose", "description": "사전 정의된 홈 포즈로 이동.", "parameters": {"type": "object", "properties": {"speed": {"type": "number"}, "wait": {"type": "boolean", "default": True}, "timeout": {"type": "number", "default": 5}}}}},
]


# ---------------- Function router ----------------

def process_function_call(function_name, arguments, robot, timestep_ms, motors, sensors, ir_sensors, touch_sensors):
    try:
        if function_name == "move_joints":
            targets = arguments.get("targets", {})
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            tolerance = arguments.get("tolerance", 0.01)
            timeout = arguments.get("timeout", 5)
            applied = {}
            for name, tgt in targets.items():
                m = motors.get(name)
                if not m:
                    continue
                if speed is not None:
                    m.setVelocity(abs(float(speed)))
                clamped = clamp_to_limits(m, float(tgt))
                m.setPosition(clamped)
                applied[name] = clamped
            if wait_flag and applied:
                reached = wait_until_targets_reached(robot, timestep_ms, sensors, applied, tolerance, timeout)
                return {"status": "ok", "applied_targets": applied, "reached": bool(reached)}
            return {"status": "ok", "applied_targets": applied}

        if function_name == "move_joint_delta":
            joint = arguments.get("joint")
            delta = float(arguments.get("delta"))
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            tolerance = arguments.get("tolerance", 0.01)
            timeout = arguments.get("timeout", 5)
            m = motors.get(joint)
            s = sensors.get(f"{joint}_sensor")
            if not m or not s:
                return {"status": "error", "message": "unknown joint"}
            if speed is not None:
                m.setVelocity(abs(float(speed)))
            target = clamp_to_limits(m, s.getValue() + delta)
            m.setPosition(target)
            if wait_flag:
                reached = wait_until_targets_reached(robot, timestep_ms, sensors, {joint: target}, tolerance, timeout)
                return {"status": "ok", "target": target, "reached": bool(reached)}
            return {"status": "ok", "target": target}

        if function_name == "set_joint_velocity_limits":
            limits = arguments.get("limits", {})
            for name, vel in limits.items():
                m = motors.get(name)
                if m:
                    m.setVelocity(abs(float(vel)))
            return {"status": "ok"}

        if function_name == "open_gripper":
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            timeout = arguments.get("timeout", 3)
            targets = {}
            for name in GRIPPER_MOTOR_NAMES:
                m = motors.get(name)
                if not m:
                    continue
                if speed is not None:
                    m.setVelocity(abs(float(speed)))
                max_pos = m.getMaxPosition()
                tgt = max_pos if max_pos is not None else 0.8
                tgt = clamp_to_limits(m, tgt)
                m.setPosition(tgt)
                targets[name] = tgt
            if wait_flag and targets:
                reached = wait_until_targets_reached(robot, timestep_ms, sensors, targets, 0.01, timeout)
                return {"status": "ok", "targets": targets, "reached": bool(reached)}
            return {"status": "ok", "targets": targets}

        if function_name == "close_gripper":
            speed = arguments.get("speed")
            stop_on_contact = arguments.get("stop_on_contact", True)
            wait_flag = arguments.get("wait", True)
            timeout = arguments.get("timeout", 3)
            targets = {}
            for name in GRIPPER_MOTOR_NAMES:
                m = motors.get(name)
                if not m:
                    continue
                if speed is not None:
                    m.setVelocity(abs(float(speed)))
                min_pos = m.getMinPosition()
                tgt = min_pos if min_pos is not None else 0.0
                tgt = clamp_to_limits(m, tgt)
                m.setPosition(tgt)
                targets[name] = tgt
            if wait_flag:
                start = time.time()
                while time.time() - start < timeout:
                    if robot.step(timestep_ms) == -1:
                        break
                    if stop_on_contact:
                        for ts in touch_sensors.values():
                            if ts.getValue() > 0.0:
                                return {"status": "ok", "targets": targets, "contact": True}
                return {"status": "ok", "targets": targets}
            return {"status": "ok", "targets": targets}

        if function_name == "set_gripper":
            left = arguments.get("left")
            right = arguments.get("right")
            both = arguments.get("both")
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            timeout = arguments.get("timeout", 3)
            to_apply = {}
            if both is not None:
                to_apply["gripper::left"] = float(both)
                to_apply["gripper::right"] = float(both)
            if left is not None:
                to_apply["gripper::left"] = float(left)
            if right is not None:
                to_apply["gripper::right"] = float(right)
            targets = {}
            for name, tgt in to_apply.items():
                m = motors.get(name)
                if not m:
                    continue
                if speed is not None:
                    m.setVelocity(abs(float(speed)))
                clamped = clamp_to_limits(m, tgt)
                m.setPosition(clamped)
                targets[name] = clamped
            if wait_flag and targets:
                reached = wait_until_targets_reached(robot, timestep_ms, sensors, targets, 0.01, timeout)
                return {"status": "ok", "targets": targets, "reached": bool(reached)}
            return {"status": "ok", "targets": targets}

        if function_name == "read_joint_positions":
            readings = {}
            for name in JOINT_MOTOR_NAMES + GRIPPER_MOTOR_NAMES:
                s = sensors.get(f"{name}_sensor")
                if s:
                    readings[name] = s.getValue()
            return {"status": "ok", "positions": readings}

        if function_name == "read_gripper_contacts":
            return {"status": "ok", "touch": {k: v.getValue() for k, v in touch_sensors.items()}}

        if function_name == "read_proximity":
            return {"status": "ok", "proximity": {k: v.getValue() for k, v in ir_sensors.items()}}

        if function_name == "wait_until_reached":
            targets = arguments.get("targets", {})
            tolerance = arguments.get("tolerance", 0.01)
            timeout = arguments.get("timeout", 5)
            reached = wait_until_targets_reached(robot, timestep_ms, sensors, targets, tolerance, timeout)
            return {"status": "ok", "reached": bool(reached)}

        if function_name == "stop_all_motors":
            for name, m in motors.items():
                s = sensors.get(f"{name}_sensor")
                if s:
                    m.setPosition(s.getValue())
                m.setVelocity(0.0)
            return {"status": "ok"}

        if function_name == "go_home_pose":
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            timeout = arguments.get("timeout", 5)
            targets = {}
            for name, angle in HOME_POSE.items():
                m = motors.get(name)
                if not m:
                    continue
                if speed is not None:
                    m.setVelocity(abs(float(speed)))
                clamped = clamp_to_limits(m, angle)
                m.setPosition(clamped)
                targets[name] = clamped
            if wait_flag and targets:
                reached = wait_until_targets_reached(robot, timestep_ms, sensors, targets, 0.01, timeout)
                return {"status": "ok", "targets": targets, "reached": bool(reached)}
            return {"status": "ok", "targets": targets}

        return {"status": "error", "message": "unknown function"}
    except Exception as e:
        return {"status": "error", "message": f"exception: {e}"}


# ---------------- LLM wrapper (Responses API 우선, Chat Completions 백업) ----------------

def handle_llm_function_calling(client, user_message, robot, timestep_ms, motors, sensors, ir_sensors, touch_sensors):
    if client is None:
        return "OpenAI 클라이언트가 초기화되지 않았습니다."

    system_prompt = (
        "너는 Webots의 IPR HD6Ms180 로봇 팔 제어 에이전트다. "
        "항상 제공된 함수(tools)만 사용해 관절/그리퍼를 제어하라."
    )

    # Responses API (공식 권장) — tool_choice="required"로 툴콜 강제
    try:
        responses_input = [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "text", "text": user_message}]},
        ]
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=responses_input,
            tools=TOOLS,
            tool_choice="required",
            max_output_tokens=1000,
            timeout=20,
        )

        # tool_use 이벤트 파싱
        tool_used = False
        if hasattr(resp, "output") and resp.output:
            for item in resp.output:
                if getattr(item, "type", None) == "tool_use":
                    fn = item.name
                    args = item.input if isinstance(item.input, dict) else {}
                    result = process_function_call(fn, args, robot, timestep_ms, motors, sensors, ir_sensors, touch_sensors)
                    tool_used = True
                    return json.dumps(result, ensure_ascii=False)
        # 툴콜이 없으면 텍스트 응답 반환 시도
        try:
            output_text = getattr(resp, "output_text", None)
            if output_text:
                return output_text
        except Exception:
            pass
        if not tool_used:
            # 강제 required인데도 툴콜이 없다면 에러 반환
            return "모델이 툴콜을 생성하지 않았습니다. 프롬프트/스키마를 확인하세요."
    except Exception:
        # 아래 Chat Completions 백업 경로 사용
        pass

    # Chat Completions (백업) — tools + tool_choice="required"
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
        cc = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="required",
            max_completion_tokens=400,
            timeout=20,
        )
        message = cc.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            call = tool_calls[0]
            fn = call.function.name
            try:
                args = json.loads(call.function.arguments) if isinstance(call.function.arguments, str) else call.function.arguments
            except Exception:
                args = {}
            result = process_function_call(fn, args, robot, timestep_ms, motors, sensors, ir_sensors, touch_sensors)
            return json.dumps(result, ensure_ascii=False)
        return message.content if getattr(message, "content", None) else "No actionable output."
    except Exception as e:
        return f"LLM 처리 오류: {e}"


# ---------------- Initialization ----------------

dotenv.load_dotenv()
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("OpenAI 클라이언트 초기화 완료 (gpt-4o-mini)")
except Exception as e:
    print(f"OpenAI 클라이언트 초기화 실패: {e}")
    client = None

robot = Robot()
timestep = int(robot.getBasicTimeStep())
print(f"기본 시간 스텝: {timestep} ms")

# Motors
motors = {}
for name in JOINT_MOTOR_NAMES + GRIPPER_MOTOR_NAMES:
    dev = None
    try:
        dev = robot.getDevice(name)
    except Exception:
        dev = None
    if dev:
        try:
            dev.setVelocity(1.0)
        except Exception:
            pass
        motors[name] = dev
    else:
        print(f"[WARN] Motor device not found: {name}")

# Position sensors
sensors = {}
for name in JOINT_SENSOR_NAMES:
    try:
        s = robot.getDevice(name)
    except Exception:
        s = None
    if s:
        try:
            s.enable(timestep)
        except Exception:
            pass
        sensors[name] = s
    else:
        print(f"[WARN] PositionSensor not found: {name}")

# IR proximity sensors
ir_sensors = {}
for name in IR_SENSOR_NAMES:
    try:
        ds = robot.getDevice(name)
    except Exception:
        ds = None
    if ds:
        try:
            ds.enable(timestep)
        except Exception:
            pass
        ir_sensors[name] = ds

# Touch sensors on gripper
touch_sensors = {}
for name in TOUCH_SENSOR_NAMES:
    try:
        ts = robot.getDevice(name)
    except Exception:
        ts = None
    if ts:
        try:
            ts.enable(timestep)
        except Exception:
            pass
        touch_sensors[name] = ts

print(f"Loaded motors: {list(motors.keys())}")
print(f"Loaded pos sensors: {list(sensors.keys())}")
print(f"Loaded IR sensors: {list(ir_sensors.keys())}")
print(f"Loaded touch sensors: {list(touch_sensors.keys())}")

if len(motors) == 0:
    print("[HINT] 팔 모터를 발견하지 못했습니다. world의 IprHd6ms180에 controller를 'vlm_controller'로 설정하세요.")


# ---------------- Main loop ----------------

while robot.step(timestep) != -1:
    # 윈도우 입력 제거: 컨트롤러는 주기 스텝만 수행
    pass

from controller import Robot
from openai import OpenAI
import os
import time
import json
import dotenv


# ---------------- Device name constants (from IprHd6ms180.proto) ----------------

JOINT_MOTOR_NAMES = [
    "base",
    "upperarm",
    "forearm",
    "wrist",
    "rotational_wrist",
]

GRIPPER_MOTOR_NAMES = [
    "gripper::left",
    "gripper::right",
]

JOINT_SENSOR_NAMES = [
    "base_sensor",
    "upperarm_sensor",
    "forearm_sensor",
    "wrist_sensor",
    "rotational_wrist_sensor",
    "gripper::left_sensor",
    "gripper::right_sensor",
]

IR_SENSOR_NAMES = [
    "ds0", "ds1", "ds2", "ds3", "ds4", "ds5", "ds6", "ds7", "ds8"
]

TOUCH_SENSOR_NAMES = [
    "ts0", "ts1", "ts2", "ts3"
]

HOME_POSE = {
    "base": 0.0,
    "upperarm": 0.0,
    "forearm": 0.0,
    "wrist": 0.0,
    "rotational_wrist": 0.0,
}


# ---------------- Helpers ----------------

def clamp_to_limits(motor, target):
    try:
        min_pos = motor.getMinPosition()
        max_pos = motor.getMaxPosition()
        if min_pos is None or max_pos is None:
            return target
        if max_pos < min_pos:
            return target
        if target < min_pos:
            return min_pos
        if target > max_pos:
            return max_pos
        return target
    except Exception:
        return target


def wait_until_targets_reached(robot, timestep_ms, sensors, targets_by_name, tolerance=0.01, timeout_s=5.0):
    start = time.time()
    while time.time() - start < timeout_s:
        all_ok = True
        for name, target in targets_by_name.items():
            sensor_name = f"{name}_sensor"
            sensor = sensors.get(sensor_name)
            if sensor is None:
                continue
            current = sensor.getValue()
            if abs(current - target) > tolerance:
                all_ok = False
                break
        if all_ok:
            return True
        if robot.step(timestep_ms) == -1:
            break
    return False


# ---------------- LLM Tool (Function) Schemas ----------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "move_joints",
            "description": "Move multiple joints to target angles (radians). Optionally wait until reached.",
            "parameters": {
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "object",
                        "additionalProperties": {"type": "number"},
                        "description": "Map joint name to rad: base, upperarm, forearm, wrist, rotational_wrist"
                    },
                    "speed": {"type": "number"},
                    "wait": {"type": "boolean", "default": True},
                    "tolerance": {"type": "number", "default": 0.01},
                    "timeout": {"type": "number", "default": 5}
                },
                "required": ["targets"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "move_joint_delta",
            "description": "Move a single joint by a relative delta (radians).",
            "parameters": {
                "type": "object",
                "properties": {
                    "joint": {"type": "string", "enum": JOINT_MOTOR_NAMES},
                    "delta": {"type": "number"},
                    "speed": {"type": "number"},
                    "wait": {"type": "boolean", "default": True},
                    "tolerance": {"type": "number", "default": 0.01},
                    "timeout": {"type": "number", "default": 5}
                },
                "required": ["joint", "delta"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_joint_velocity_limits",
            "description": "Set velocity limits (rad/s) for specified joints.",
            "parameters": {
                "type": "object",
                "properties": {
                    "limits": {"type": "object", "additionalProperties": {"type": "number"}}
                },
                "required": ["limits"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "open_gripper",
            "description": "Open the gripper to maximum aperture.",
            "parameters": {"type": "object", "properties": {"speed": {"type": "number"}, "wait": {"type": "boolean", "default": True}, "timeout": {"type": "number", "default": 3}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "close_gripper",
            "description": "Close the gripper. Optionally stop when contact detected.",
            "parameters": {"type": "object", "properties": {"speed": {"type": "number"}, "stop_on_contact": {"type": "boolean", "default": True}, "wait": {"type": "boolean", "default": True}, "timeout": {"type": "number", "default": 3}}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_gripper",
            "description": "Set gripper finger positions (radians). If 'both' is set, applies to both fingers.",
            "parameters": {"type": "object", "properties": {"left": {"type": "number"}, "right": {"type": "number"}, "both": {"type": "number"}, "speed": {"type": "number"}, "wait": {"type": "boolean", "default": True}, "timeout": {"type": "number", "default": 3}}}
        }
    },
    {"type": "function", "function": {"name": "read_joint_positions", "description": "Read all joint and gripper angles (rad).", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_gripper_contacts", "description": "Read touch sensors ts0..ts3.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "read_proximity", "description": "Read IR proximity sensors ds0..ds8.", "parameters": {"type": "object", "properties": {}}}},
    {
        "type": "function",
        "function": {
            "name": "wait_until_reached",
            "description": "Wait until specified joints reach targets within tolerance.",
            "parameters": {"type": "object", "properties": {"targets": {"type": "object", "additionalProperties": {"type": "number"}}, "tolerance": {"type": "number", "default": 0.01}, "timeout": {"type": "number", "default": 5}}, "required": ["targets"]}
        }
    },
    {"type": "function", "function": {"name": "stop_all_motors", "description": "Stop all motors and hold current positions.", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "go_home_pose", "description": "Move joints to predefined home pose.", "parameters": {"type": "object", "properties": {"speed": {"type": "number"}, "wait": {"type": "boolean", "default": True}, "timeout": {"type": "number", "default": 5}}}}},
]


# ---------------- Function router ----------------

def process_function_call(function_name, arguments, robot, timestep_ms, motors, sensors, ir_sensors, touch_sensors):
    try:
        if function_name == "move_joints":
            targets = arguments.get("targets", {})
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            tolerance = arguments.get("tolerance", 0.01)
            timeout = arguments.get("timeout", 5)
            applied = {}
            for name, tgt in targets.items():
                m = motors.get(name)
                if not m:
                    continue
                if speed is not None:
                    m.setVelocity(abs(float(speed)))
                clamped = clamp_to_limits(m, float(tgt))
                m.setPosition(clamped)
                applied[name] = clamped
            if wait_flag and applied:
                reached = wait_until_targets_reached(robot, timestep_ms, sensors, applied, tolerance, timeout)
                return {"status": "ok", "applied_targets": applied, "reached": bool(reached)}
            return {"status": "ok", "applied_targets": applied}

        if function_name == "move_joint_delta":
            joint = arguments.get("joint")
            delta = float(arguments.get("delta"))
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            tolerance = arguments.get("tolerance", 0.01)
            timeout = arguments.get("timeout", 5)
            m = motors.get(joint)
            s = sensors.get(f"{joint}_sensor")
            if not m or not s:
                return {"status": "error", "message": "unknown joint"}
            if speed is not None:
                m.setVelocity(abs(float(speed)))
            target = clamp_to_limits(m, s.getValue() + delta)
            m.setPosition(target)
            if wait_flag:
                reached = wait_until_targets_reached(robot, timestep_ms, sensors, {joint: target}, tolerance, timeout)
                return {"status": "ok", "target": target, "reached": bool(reached)}
            return {"status": "ok", "target": target}

        if function_name == "set_joint_velocity_limits":
            limits = arguments.get("limits", {})
            for name, vel in limits.items():
                m = motors.get(name)
                if m:
                    m.setVelocity(abs(float(vel)))
            return {"status": "ok"}

        if function_name == "open_gripper":
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            timeout = arguments.get("timeout", 3)
            targets = {}
            for name in GRIPPER_MOTOR_NAMES:
                m = motors.get(name)
                if not m:
                    continue
                if speed is not None:
                    m.setVelocity(abs(float(speed)))
                max_pos = m.getMaxPosition()
                tgt = max_pos if max_pos is not None else 0.8
                tgt = clamp_to_limits(m, tgt)
                m.setPosition(tgt)
                targets[name] = tgt
            if wait_flag and targets:
                reached = wait_until_targets_reached(robot, timestep_ms, sensors, targets, 0.01, timeout)
                return {"status": "ok", "targets": targets, "reached": bool(reached)}
            return {"status": "ok", "targets": targets}

        if function_name == "close_gripper":
            speed = arguments.get("speed")
            stop_on_contact = arguments.get("stop_on_contact", True)
            wait_flag = arguments.get("wait", True)
            timeout = arguments.get("timeout", 3)
            targets = {}
            for name in GRIPPER_MOTOR_NAMES:
                m = motors.get(name)
                if not m:
                    continue
                if speed is not None:
                    m.setVelocity(abs(float(speed)))
                min_pos = m.getMinPosition()
                tgt = min_pos if min_pos is not None else 0.0
                tgt = clamp_to_limits(m, tgt)
                m.setPosition(tgt)
                targets[name] = tgt
            if wait_flag:
                start = time.time()
                while time.time() - start < timeout:
                    if robot.step(timestep_ms) == -1:
                        break
                    if stop_on_contact:
                        for ts in touch_sensors.values():
                            if ts.getValue() > 0.0:
                                return {"status": "ok", "targets": targets, "contact": True}
                return {"status": "ok", "targets": targets}
            return {"status": "ok", "targets": targets}

        if function_name == "set_gripper":
            left = arguments.get("left")
            right = arguments.get("right")
            both = arguments.get("both")
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            timeout = arguments.get("timeout", 3)
            to_apply = {}
            if both is not None:
                to_apply["gripper::left"] = float(both)
                to_apply["gripper::right"] = float(both)
            if left is not None:
                to_apply["gripper::left"] = float(left)
            if right is not None:
                to_apply["gripper::right"] = float(right)
            targets = {}
            for name, tgt in to_apply.items():
                m = motors.get(name)
                if not m:
                    continue
                if speed is not None:
                    m.setVelocity(abs(float(speed)))
                clamped = clamp_to_limits(m, tgt)
                m.setPosition(clamped)
                targets[name] = clamped
            if wait_flag and targets:
                reached = wait_until_targets_reached(robot, timestep_ms, sensors, targets, 0.01, timeout)
                return {"status": "ok", "targets": targets, "reached": bool(reached)}
            return {"status": "ok", "targets": targets}

        if function_name == "read_joint_positions":
            readings = {}
            for name in JOINT_MOTOR_NAMES + GRIPPER_MOTOR_NAMES:
                s = sensors.get(f"{name}_sensor")
                if s:
                    readings[name] = s.getValue()
            return {"status": "ok", "positions": readings}

        if function_name == "read_gripper_contacts":
            return {"status": "ok", "touch": {k: v.getValue() for k, v in touch_sensors.items()}}

        if function_name == "read_proximity":
            return {"status": "ok", "proximity": {k: v.getValue() for k, v in ir_sensors.items()}}

        if function_name == "wait_until_reached":
            targets = arguments.get("targets", {})
            tolerance = arguments.get("tolerance", 0.01)
            timeout = arguments.get("timeout", 5)
            reached = wait_until_targets_reached(robot, timestep_ms, sensors, targets, tolerance, timeout)
            return {"status": "ok", "reached": bool(reached)}

        if function_name == "stop_all_motors":
            for name, m in motors.items():
                s = sensors.get(f"{name}_sensor")
                if s:
                    m.setPosition(s.getValue())
                m.setVelocity(0.0)
            return {"status": "ok"}

        if function_name == "go_home_pose":
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            timeout = arguments.get("timeout", 5)
            targets = {}
            for name, angle in HOME_POSE.items():
                m = motors.get(name)
                if not m:
                    continue
                if speed is not None:
                    m.setVelocity(abs(float(speed)))
                clamped = clamp_to_limits(m, angle)
                m.setPosition(clamped)
                targets[name] = clamped
            if wait_flag and targets:
                reached = wait_until_targets_reached(robot, timestep_ms, sensors, targets, 0.01, timeout)
                return {"status": "ok", "targets": targets, "reached": bool(reached)}
            return {"status": "ok", "targets": targets}

        return {"status": "error", "message": "unknown function"}
    except Exception as e:
        return {"status": "error", "message": f"exception: {e}"}


# ---------------- LLM wrapper (Chat Completions + tools per official docs) ----------------

def handle_llm_function_calling(client, user_message, robot, timestep_ms, motors, sensors, ir_sensors, touch_sensors):
    if client is None:
        return "OpenAI client unavailable"

    system_prompt = (
        "You control a 6-DOF IPR HD6Ms180 robotic arm in Webots. "
        "Use the provided functions (tools) to move joints and the gripper."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    # Reasoning-capable, cost-effective model
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        max_completion_tokens=400,
        timeout=20,
    )

    message = response.choices[0].message
    # Tool calls per official Chat Completions tools interface
    tool_calls = getattr(message, "tool_calls", None)
    print(tool_calls)
    if tool_calls:
        # Execute the first tool call only (chain-of-tools can be added later)
        call = tool_calls[0]
        fn = call.function.name
        try:
            args = json.loads(call.function.arguments) if isinstance(call.function.arguments, str) else call.function.arguments
        except Exception:
            args = {}
        result = process_function_call(fn, args, robot, timestep_ms, motors, sensors, ir_sensors, touch_sensors)
        return json.dumps(result, ensure_ascii=False)

    return message.content if getattr(message, "content", None) else "No actionable output."


# ---------------- Initialization ----------------

dotenv.load_dotenv()
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("OpenAI client initialized (gpt-4o-mini)")
except Exception as e:
    print(f"OpenAI client init failed: {e}")
    client = None

robot = Robot()
timestep = int(robot.getBasicTimeStep())
print(f"Basic timestep: {timestep} ms")

# Motors
motors = {}
for name in JOINT_MOTOR_NAMES + GRIPPER_MOTOR_NAMES:
    dev = None
    try:
        dev = robot.getDevice(name)
    except Exception:
        dev = None
    if dev:
        try:
            dev.setVelocity(1.0)
        except Exception:
            pass
        motors[name] = dev
    else:
        print(f"[WARN] Motor device not found: {name}")

# Position sensors
sensors = {}
for name in JOINT_SENSOR_NAMES:
    try:
        s = robot.getDevice(name)
    except Exception:
        s = None
    if s:
        try:
            s.enable(timestep)
        except Exception:
            pass
        sensors[name] = s
    else:
        print(f"[WARN] PositionSensor not found: {name}")

# IR proximity sensors
ir_sensors = {}
for name in IR_SENSOR_NAMES:
    try:
        ds = robot.getDevice(name)
    except Exception:
        ds = None
    if ds:
        try:
            ds.enable(timestep)
        except Exception:
            pass
        ir_sensors[name] = ds

# Touch sensors on gripper
touch_sensors = {}
for name in TOUCH_SENSOR_NAMES:
    try:
        ts = robot.getDevice(name)
    except Exception:
        ts = None
    if ts:
        try:
            ts.enable(timestep)
        except Exception:
            pass
        touch_sensors[name] = ts

print(f"Loaded motors: {list(motors.keys())}")
print(f"Loaded pos sensors: {list(sensors.keys())}")
print(f"Loaded IR sensors: {list(ir_sensors.keys())}")
print(f"Loaded touch sensors: {list(touch_sensors.keys())}")

if len(motors) == 0:
    print("[HINT] No arm motors detected. Ensure the IprHd6ms180 node has controller set to 'vlm_controller'.")


# ---------------- Main loop ----------------

# while robot.step(timestep) != -1:
    # msg = robot.wwiReceiveText()
msg = '팔을 움직여'
# if not msg:
#     continue
print("USER_MESSAGE:", msg)
try:
    result = handle_llm_function_calling(client, msg, robot, timestep, motors, sensors, ir_sensors, touch_sensors)
except Exception as e:
    result = f"Error: {e}"
    # robot.wwiSendText(result)


