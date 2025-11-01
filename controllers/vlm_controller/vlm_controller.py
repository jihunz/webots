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
    min_pos = motor.getMinPosition()
    max_pos = motor.getMaxPosition()
    if min_pos is None or max_pos is None:
        return target
    if max_pos < min_pos:
        # Some motors may not provide limits; guard anyway
        return target
    if target < min_pos:
        return min_pos
    if target > max_pos:
        return max_pos
    return target


def wait_until_targets_reached(robot, timestep_ms, motors, sensors, targets_by_name, tolerance=0.01, timeout_s=5.0):
    start = time.time()
    name_to_sensor = {s.getName(): s for s in sensors.values()}
    while time.time() - start < timeout_s:
        all_ok = True
        for name, target in targets_by_name.items():
            sensor_name = f"{name}_sensor" if not name.startswith("gripper::") else f"{name}_sensor"
            sensor = name_to_sensor.get(sensor_name)
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


# ---------------- Function Calling schema ----------------

JOINT_ENUM = JOINT_MOTOR_NAMES

functions = [
    {
        "name": "move_joints",
        "description": "Move multiple joints to target angles (radians). Optionally wait until reached.",
        "parameters": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "number"
                    },
                    "description": "Map of joint name to target rad. Keys among: base, upperarm, forearm, wrist, rotational_wrist"
                },
                "speed": {"type": "number", "description": "rad/s speed limit for all specified joints"},
                "acceleration": {"type": "number", "description": "rad/s^2 acceleration limit (if supported)"},
                "wait": {"type": "boolean", "default": True},
                "tolerance": {"type": "number", "default": 0.01},
                "timeout": {"type": "number", "default": 5}
            },
            "required": ["targets"]
        }
    },
    {
        "name": "move_joint_delta",
        "description": "Move a single joint by a delta (radians) relative to current.",
        "parameters": {
            "type": "object",
            "properties": {
                "joint": {"type": "string", "enum": JOINT_ENUM},
                "delta": {"type": "number"},
                "speed": {"type": "number"},
                "wait": {"type": "boolean", "default": True},
                "tolerance": {"type": "number", "default": 0.01},
                "timeout": {"type": "number", "default": 5}
            },
            "required": ["joint", "delta"]
        }
    },
    {
        "name": "set_joint_velocity_limits",
        "description": "Set velocity limits for joints (rad/s). Applies to position-control speed cap.",
        "parameters": {
            "type": "object",
            "properties": {
                "limits": {
                    "type": "object",
                    "additionalProperties": {"type": "number"},
                    "description": "Map of joint name to velocity rad/s"
                }
            },
            "required": ["limits"]
        }
    },
    {
        "name": "open_gripper",
        "description": "Open the gripper to its maximum aperture.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {"type": "number"},
                "wait": {"type": "boolean", "default": True},
                "timeout": {"type": "number", "default": 3}
            }
        }
    },
    {
        "name": "close_gripper",
        "description": "Close the gripper. Optionally stop when contact is detected.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {"type": "number"},
                "stop_on_contact": {"type": "boolean", "default": True},
                "wait": {"type": "boolean", "default": True},
                "timeout": {"type": "number", "default": 3}
            }
        }
    },
    {
        "name": "set_gripper",
        "description": "Set gripper finger positions (radians). If 'both' given, applies to left and right.",
        "parameters": {
            "type": "object",
            "properties": {
                "left": {"type": "number"},
                "right": {"type": "number"},
                "both": {"type": "number"},
                "speed": {"type": "number"},
                "wait": {"type": "boolean", "default": True},
                "timeout": {"type": "number", "default": 3}
            }
        }
    },
    {
        "name": "read_joint_positions",
        "description": "Read all joint and gripper angles (radians).",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "read_gripper_contacts",
        "description": "Read touch sensor values ts0..ts3.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "read_proximity",
        "description": "Read IR proximity sensors ds0..ds8.",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "wait_until_reached",
        "description": "Wait until specified joints reach targets within tolerance.",
        "parameters": {
            "type": "object",
            "properties": {
                "targets": {
                    "type": "object",
                    "additionalProperties": {"type": "number"}
                },
                "tolerance": {"type": "number", "default": 0.01},
                "timeout": {"type": "number", "default": 5}
            },
            "required": ["targets"]
        }
    },
    {
        "name": "stop_all_motors",
        "description": "Stop all motors safely (hold current positions).",
        "parameters": {"type": "object", "properties": {}}
    },
    {
        "name": "go_home_pose",
        "description": "Move all joints to a predefined home pose.",
        "parameters": {
            "type": "object",
            "properties": {
                "speed": {"type": "number"},
                "wait": {"type": "boolean", "default": True},
                "timeout": {"type": "number", "default": 5}
            }
        }
    },
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

            applied_targets = {}
            for name, target in targets.items():
                motor = motors.get(name)
                if motor is None:
                    continue
                if speed is not None:
                    motor.setVelocity(abs(speed))
                clamped = clamp_to_limits(motor, float(target))
                motor.setPosition(clamped)
                applied_targets[name] = clamped

            if wait_flag and applied_targets:
                reached = wait_until_targets_reached(robot, timestep_ms, motors, sensors, applied_targets, tolerance, timeout)
                return {"status": "ok", "applied_targets": applied_targets, "reached": bool(reached)}
            return {"status": "ok", "applied_targets": applied_targets}

        if function_name == "move_joint_delta":
            joint = arguments.get("joint")
            delta = float(arguments.get("delta"))
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            tolerance = arguments.get("tolerance", 0.01)
            timeout = arguments.get("timeout", 5)

            motor = motors.get(joint)
            sensor = sensors.get(f"{joint}_sensor")
            if motor is None or sensor is None:
                return {"status": "error", "message": "unknown joint"}
            if speed is not None:
                motor.setVelocity(abs(speed))
            current = sensor.getValue()
            target = clamp_to_limits(motor, current + delta)
            motor.setPosition(target)
            if wait_flag:
                reached = wait_until_targets_reached(robot, timestep_ms, motors, sensors, {joint: target}, tolerance, timeout)
                return {"status": "ok", "target": target, "reached": bool(reached)}
            return {"status": "ok", "target": target}

        if function_name == "set_joint_velocity_limits":
            limits = arguments.get("limits", {})
            for name, vel in limits.items():
                motor = motors.get(name)
                if motor is None:
                    continue
                motor.setVelocity(abs(float(vel)))
            return {"status": "ok"}

        if function_name == "open_gripper":
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            timeout = arguments.get("timeout", 3)
            targets = {}
            for name in GRIPPER_MOTOR_NAMES:
                m = motors.get(name)
                if m is None:
                    continue
                if speed is not None:
                    m.setVelocity(abs(speed))
                max_pos = m.getMaxPosition()
                target = max_pos if max_pos is not None else 0.8
                target = clamp_to_limits(m, target)
                m.setPosition(target)
                targets[name] = target
            if wait_flag and targets:
                reached = wait_until_targets_reached(robot, timestep_ms, motors, sensors, targets, 0.01, timeout)
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
                if m is None:
                    continue
                if speed is not None:
                    m.setVelocity(abs(speed))
                min_pos = m.getMinPosition()
                target = min_pos if min_pos is not None else 0.0
                target = clamp_to_limits(m, target)
                m.setPosition(target)
                targets[name] = target

            if wait_flag:
                start = time.time()
                while time.time() - start < timeout:
                    if robot.step(timestep_ms) == -1:
                        break
                    if stop_on_contact:
                        # If any touch sensor detects force, stop waiting early
                        contacted = False
                        for ts in touch_sensors.values():
                            if ts.getValue() > 0.0:
                                contacted = True
                                break
                        if contacted:
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
            targets = {}
            assignments = {}
            if both is not None:
                assignments["gripper::left"] = float(both)
                assignments["gripper::right"] = float(both)
            if left is not None:
                assignments["gripper::left"] = float(left)
            if right is not None:
                assignments["gripper::right"] = float(right)
            for name, tgt in assignments.items():
                m = motors.get(name)
                if m is None:
                    continue
                if speed is not None:
                    m.setVelocity(abs(speed))
                clamped = clamp_to_limits(m, tgt)
                m.setPosition(clamped)
                targets[name] = clamped
            if wait_flag and targets:
                reached = wait_until_targets_reached(robot, timestep_ms, motors, sensors, targets, 0.01, timeout)
                return {"status": "ok", "targets": targets, "reached": bool(reached)}
            return {"status": "ok", "targets": targets}

        if function_name == "read_joint_positions":
            readings = {}
            for name in JOINT_MOTOR_NAMES + GRIPPER_MOTOR_NAMES:
                s = sensors.get(f"{name}_sensor")
                if s is not None:
                    readings[name] = s.getValue()
            return {"status": "ok", "positions": readings}

        if function_name == "read_gripper_contacts":
            readings = {name: ts.getValue() for name, ts in touch_sensors.items()}
            return {"status": "ok", "touch": readings}

        if function_name == "read_proximity":
            readings = {name: ds.getValue() for name, ds in ir_sensors.items()}
            return {"status": "ok", "proximity": readings}

        if function_name == "wait_until_reached":
            targets = arguments.get("targets", {})
            tolerance = arguments.get("tolerance", 0.01)
            timeout = arguments.get("timeout", 5)
            reached = wait_until_targets_reached(robot, timestep_ms, motors, sensors, targets, tolerance, timeout)
            return {"status": "ok", "reached": bool(reached)}

        if function_name == "stop_all_motors":
            # Hold current positions
            for name, motor in motors.items():
                sensor = sensors.get(f"{name}_sensor")
                if sensor is not None:
                    motor.setPosition(sensor.getValue())
                motor.setVelocity(0.0)
            return {"status": "ok"}

        if function_name == "go_home_pose":
            speed = arguments.get("speed")
            wait_flag = arguments.get("wait", True)
            timeout = arguments.get("timeout", 5)
            targets = {}
            for name, angle in HOME_POSE.items():
                m = motors.get(name)
                if m is None:
                    continue
                if speed is not None:
                    m.setVelocity(abs(speed))
                clamped = clamp_to_limits(m, angle)
                m.setPosition(clamped)
                targets[name] = clamped
            if wait_flag:
                reached = wait_until_targets_reached(robot, timestep_ms, motors, sensors, targets, 0.01, timeout)
                return {"status": "ok", "targets": targets, "reached": bool(reached)}
            return {"status": "ok", "targets": targets}

        return {"status": "error", "message": "unknown function"}
    except Exception as e:
        return {"status": "error", "message": f"exception: {e}"}


# ---------------- LLM wrapper ----------------

def handle_llm_function_calling(client, user_message, robot, timestep_ms, motors, sensors, ir_sensors, touch_sensors):
    if client is None:
        return "OpenAI client unavailable"

    system_prompt = (
        "You control a 6-DOF IPR HD6Ms180 robotic arm in Webots. "
        "Always use the provided functions to move joints and the gripper. "
        "Prefer move_joints for multi-joint moves; use read_* to perceive before acting."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        functions=functions,
        function_call="auto",
        max_completion_tokens=400,
        timeout=20,
    )

    message = response.choices[0].message
    if hasattr(message, "function_call") and message.function_call:
        fn = message.function_call.name
        args = json.loads(message.function_call.arguments)
        result = process_function_call(fn, args, robot, timestep_ms, motors, sensors, ir_sensors, touch_sensors)
        return json.dumps(result, ensure_ascii=False)

    return message.content if getattr(message, "content", None) else "No actionable output."


# ---------------- Initialization ----------------

dotenv.load_dotenv()
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("OpenAI client initialized")
except Exception as e:
    print(f"OpenAI client init failed: {e}")
    client = None

robot = Robot()
timestep = int(robot.getBasicTimeStep())
print(f"Basic timestep: {timestep} ms")

# Motors
motors = {}
for name in JOINT_MOTOR_NAMES + GRIPPER_MOTOR_NAMES:
    try:
        device = robot.getDevice(name)
        if device:
            # Set a reasonable default speed cap
            device.setVelocity(1.0)
            motors[name] = device
    except Exception:
        pass

# Position sensors
sensors = {}
for name in JOINT_SENSOR_NAMES:
    try:
        s = robot.getDevice(name)
        if s:
            s.enable(timestep)
            sensors[name] = s
    except Exception:
        pass

# IR proximity sensors
ir_sensors = {}
for name in IR_SENSOR_NAMES:
    try:
        ds = robot.getDevice(name)
        if ds:
            ds.enable(timestep)
            ir_sensors[name] = ds
    except Exception:
        pass

# Touch sensors on gripper
touch_sensors = {}
for name in TOUCH_SENSOR_NAMES:
    try:
        ts = robot.getDevice(name)
        if ts:
            ts.enable(timestep)
            touch_sensors[name] = ts
    except Exception:
        pass

print(f"Loaded motors: {list(motors.keys())}")
print(f"Loaded sensors: {list(sensors.keys())}")
print(f"Loaded IR: {list(ir_sensors.keys())}")
print(f"Loaded touch: {list(touch_sensors.keys())}")


# ---------------- Main loop ----------------

# while robot.step(timestep) != -1:
    # msg = robot.wwiReceiveText()
msg = "팔을 움직여서 사과를 들어올려"

# if not msg:
#     continue

print("USER_MESSAGE:", msg)
try:
    result = handle_llm_function_calling(client, msg, robot, timestep, motors, sensors, ir_sensors, touch_sensors)
except Exception as e:
    result = f"Error: {e}"

# robot.wwiSendText(result)


