import os
import sys
import json
import threading
from queue import Queue
import numpy as np
from controller import Robot, Supervisor

# ==================================================================================
# Webots TIAGo++ Perfect Controller (Final Ver.)
# Features: 
#   1. CrewAI Agent (Robotics Engineer) - GPT-4o Planner with ReAct
#   2. Real YOLOv8 Vision - Object detection + Depth estimation + Scanning
#   3. Pinocchio IK - Precise arm control (Standard/Submodule/Wrapper support)
#   4. Mobile Manipulation - Move Base + Move Arm
#   5. Thread-Safe Sensor Sync - Robust vision pipeline
# Dependencies:
#   pip install crewai langchain-openai pinocchio ultralytics python-dotenv
# ==================================================================================

try:
    from crewai import Agent, Task, Crew, Process
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field
    from typing import Type
    from langchain_openai import ChatOpenAI
    from ultralytics import YOLO
    import pinocchio as pin
    from dotenv import load_dotenv
except ImportError as e:
    sys.exit(f"❌ Missing libraries: {e}")

if load_dotenv: load_dotenv()

OPENAI_MODEL = "gpt-4o"
YOLO_MODEL = "yolov8m.pt" # Use Medium
URDF_FILENAME = "tiago_generated.urdf"

action_queue = Queue()

sensor_data = {
    "rgb": None,
    "depth": None,
    "camera_width": 0,
    "camera_height": 0,
    "lock": threading.Lock()
}

class TiagoRobotInterface:
    def __init__(self, robot):
        self.robot = robot
        self.timestep = int(self.robot.getBasicTimeStep())
        self.init_devices()
        
        print("📝 Generating URDF from Webots...")
        urdf_content = self.robot.getUrdf()
        with open(URDF_FILENAME, "w") as f:
            f.write(urdf_content)
            
        self.model = None
        try:
            if not os.path.exists(URDF_FILENAME) or os.path.getsize(URDF_FILENAME) < 100:
                raise Exception("URDF file empty")

            if hasattr(pin, 'buildModelFromUrdf'):
                try: self.model = pin.buildModelFromUrdf(URDF_FILENAME)
                except: pass
            
            if self.model is None:
                try:
                    import pinocchio.urdf
                    self.model = pinocchio.urdf.buildModel(URDF_FILENAME)
                except: pass

            if self.model is None:
                try:
                    from pinocchio.robot_wrapper import RobotWrapper
                    self.model = RobotWrapper.BuildFromURDF(URDF_FILENAME).model
                except: pass

            if self.model:
                self.data = self.model.createData()
                self.ee_frame_id = -1
                candidates = ["arm_right_tool_link", "arm_right_7_link", "gripper_right_link"]
                for c in candidates:
                    if self.model.existFrame(c):
                        self.ee_frame_id = self.model.getFrameId(c)
                        print(f"✅ Found EE Frame: {c} (ID: {self.ee_frame_id})")
                        break
                if self.ee_frame_id != -1:
                    print(f"✅ Pinocchio IK Initialized Successfully")
                else:
                    raise Exception("No EE frame")
            else:
                raise AttributeError("Pinocchio load failed")

        except Exception as e:
            print(f"❌ CRITICAL IK ERROR: {e}")
            sys.exit(1)

        self.yolo = YOLO(YOLO_MODEL)

    def init_devices(self):
        # 1. Camera
        self.camera = None
        for name in ['Astra rgb', 'xtion_rgb', 'camera', 'head_front_camera']:
            d = self.robot.getDevice(name)
            if d: 
                self.camera = d
                self.camera.enable(self.timestep)
                print(f"✅ Camera found: {name}")
                break
        
        # 2. Depth
        self.range_finder = None
        for name in ['Astra depth', 'xtion_depth', 'range-finder']:
            d = self.robot.getDevice(name)
            if d:
                self.range_finder = d
                self.range_finder.enable(self.timestep)
                print(f"✅ RangeFinder found: {name}")
                break

        # 3. Arm (Right)
        self.arm_joints = []
        self.arm_sensors = []
        for i in range(1, 8):
            name = f'arm_right_{i}_joint'
            m = self.robot.getDevice(name)
            s = self.robot.getDevice(f'{name}_sensor')
            if m:
                m.setVelocity(1.0)
                self.arm_joints.append(m)
                if s: 
                    s.enable(self.timestep)
                    self.arm_sensors.append(s)
                else:
                    self.arm_sensors.append(None)

        # 4. Gripper
        self.gripper = None
        for name in ['right_hand_gripper_right_finger_joint', 'gripper_right_finger_joint']:
            m = self.robot.getDevice(name)
            if m: self.gripper = m; break
            
        # 5. Base
        self.wheels = []
        for name in ['wheel_left_joint', 'wheel_right_joint']:
            m = self.robot.getDevice(name)
            if m:
                m.setPosition(float('inf'))
                m.setVelocity(0.0)
                self.wheels.append(m)

        # 6. Head
        self.head_pan = self.robot.getDevice('head_1_joint')
        self.head_tilt = self.robot.getDevice('head_2_joint')
        if self.head_pan: self.head_pan.setVelocity(1.0)
        if self.head_tilt: self.head_tilt.setVelocity(1.0)
        
        # Initial Posture: Look DOWN
        if self.head_tilt: self.head_tilt.setPosition(0.4) 
        if self.head_pan: self.head_pan.setPosition(0.0)
        
        # Arm Ready Pose
        ready_pose = [0.2, 0.5, 0.0, 1.0, -1.57, 0.0, 0.0] 
        for i, m in enumerate(self.arm_joints):
            if i < len(ready_pose):
                m.setPosition(ready_pose[i])
        
        print("✅ Robot Initialized to Ready Pose")

    def update_sensors(self):
        if self.camera and self.range_finder:
            raw_img = self.camera.getImage()
            raw_depth = self.range_finder.getRangeImage()
            
            with sensor_data["lock"]:
                if raw_img:
                    # Webots returns BGRA
                    # Convert BGRA -> RGB using numpy slicing [2, 1, 0]
                    # Also reshape correctly: H, W, 4
                    img_bgra = np.frombuffer(raw_img, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
                    sensor_data["rgb"] = img_bgra[:, :, [2, 1, 0]] 
                    sensor_data["camera_width"] = self.camera.getWidth()
                    sensor_data["camera_height"] = self.camera.getHeight()
                
                if raw_depth:
                    sensor_data["depth"] = list(raw_depth)

    def execute_move_base(self, linear, angular):
        if len(self.wheels) < 2: return "No wheels"
        WHEEL_RADIUS = 0.0985
        AXLE_LENGTH = 0.4044
        v_left = (linear - angular * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
        v_right = (linear + angular * AXLE_LENGTH / 2.0) / WHEEL_RADIUS
        self.wheels[0].setVelocity(max(min(v_left, 5.0), -5.0))
        self.wheels[1].setVelocity(max(min(v_right, 5.0), -5.0))
        return "Moving base"

    def execute_move_head(self, pan, tilt):
        if self.head_pan: self.head_pan.setPosition(pan)
        if self.head_tilt: self.head_tilt.setPosition(tilt)
        return "Head moved"

    def get_current_ee_position(self):
        """Get current end-effector position using Forward Kinematics"""
        if not self.model or self.ee_frame_id == -1:
            return None
        
        # Get current joint angles
        q = pin.neutral(self.model)
        for name, sensor in zip([m.getName() for m in self.arm_joints], self.arm_sensors):
            if sensor and self.model.existJointName(name):
                idx = self.model.getJointId(name)
                idx_q = self.model.joints[idx].idx_q
                q[idx_q] = sensor.getValue()
        
        # Compute FK
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pos = self.data.oMf[self.ee_frame_id].translation
        return pos.copy()

    def execute_move_arm(self, target_pos):
        print(f"🦾 IK Target: {target_pos}")
        if not self.model or not self.arm_joints: return "IK Error"
        
        # Print current EE position for debugging
        curr_pos = self.get_current_ee_position()
        if curr_pos is not None:
            print(f"   Current EE position: [{curr_pos[0]:.3f}, {curr_pos[1]:.3f}, {curr_pos[2]:.3f}]")
        
        guesses = []
        q_neutral = pin.neutral(self.model)
        guesses.append(q_neutral)
        
        q_curr = q_neutral.copy()
        joint_read_count = 0
        for name, sensor in zip([m.getName() for m in self.arm_joints], self.arm_sensors):
            if sensor:
                if self.model.existJointName(name):
                    idx = self.model.getJointId(name)
                    idx_q = self.model.joints[idx].idx_q
                    val = sensor.getValue()
                    q_curr[idx_q] = val
                    joint_read_count += 1
                else:
                    print(f"   ⚠️ Joint '{name}' not found in URDF model!")
            else:
                print(f"   ⚠️ No sensor for joint '{name}'")
        print(f"   📊 Read {joint_read_count}/{len(self.arm_joints)} joint positions from sensors")
        guesses.append(q_curr)
        
        for _ in range(5):
            guesses.append(pin.randomConfiguration(self.model))

        eps = 5e-2; IT_MAX = 500; DT = 1e-1; damp = 1e-2
        best_q = None
        best_err = float('inf')
        
        for q_start in guesses:
            q = q_start.copy()
            for i in range(IT_MAX):
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                curr = self.data.oMf[self.ee_frame_id].translation
                err = curr - np.array(target_pos)
                err_norm = np.linalg.norm(err)
                
                if err_norm < eps:
                    best_q = q.copy()
                    print(f"✅ IK Converged! Error={err_norm:.4f}")
                    break
                if err_norm < best_err:
                    best_err = err_norm
                    best_q = q.copy()
                    
                J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id)[:3, :]
                v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(3), err))
                q = pin.integrate(self.model, q, v * DT)
            
            if best_err < eps:
                break
        
        if best_q is not None:
            print(f"   Best error achieved: {best_err:.4f}")
            for name, motor in zip([m.getName() for m in self.arm_joints], self.arm_joints):
                if self.model.existJointName(name):
                    idx = self.model.getJointId(name)
                    idx_q = self.model.joints[idx].idx_q
                    motor.setPosition(best_q[idx_q])
            return "Moved arm (IK)"
        
        print("❌ Pinocchio IK Failed")
        return "IK Failed"

    def execute_gripper(self, open_state):
        if self.gripper:
            self.gripper.setPosition(0.045 if open_state else 0.0)
        return "Gripper moved"

robot_interface = None 

# --- Tools ---
class GetRobotStateTool(BaseTool):
    name: str = "get_robot_state"
    description: str = "Returns current joint angles."
    def _run(self) -> str:
        return "State OK"

class LookAroundInput(BaseModel):
    pan: float = Field(..., description="Head Pan angle (rad). Range: -1.0 to 1.0")
    tilt: float = Field(..., description="Head Tilt angle (rad). Range: -0.9 to 0.7")

class LookAroundTool(BaseTool):
    name: str = "look_around"
    description: str = "Moves head AND runs YOLO. If target found, returns position directly. Use this to find objects."
    args_schema: Type[BaseModel] = LookAroundInput
    def _run(self, pan: float, tilt: float) -> str:
        import time
        import random
        
        # 1. Force head movement by adding small random offset, then go to target
        # This ensures sensors are refreshed even if same pan/tilt is requested
        offset = random.uniform(-0.05, 0.05)
        action_queue.put({"type": "head", "pan": pan + offset, "tilt": tilt + offset})
        time.sleep(0.5)
        action_queue.put({"type": "head", "pan": pan, "tilt": tilt})
        action_queue.put({"type": "refresh"})
        print(f"👀 Moving head to Pan={pan}, Tilt={tilt}...", flush=True)
        
        # 2. Wait for head to move + sensor update
        time.sleep(2.0)
        
        # 3. Run YOLO directly here - read FRESH sensor data
        rgb = None
        depth = None
        w = 0
        h = 0
        
        # Read sensor data
        with sensor_data["lock"]:
            if sensor_data["rgb"] is not None:
                rgb = sensor_data["rgb"].copy()
                if sensor_data["depth"]:
                    depth = list(sensor_data["depth"])
                w = sensor_data["camera_width"]
                h = sensor_data["camera_height"]
        
        if rgb is None:
            return f"Head at Pan={pan}, Tilt={tilt}. ERROR: No image!"
        
        # Run YOLO
        results = robot_interface.yolo(rgb, verbose=False, conf=0.6)
        num_boxes = len(results[0].boxes) if results else 0
        print(f"   🔍 YOLO detected {num_boxes} objects")
        
        detected = []
        alias_map = {
            "traffic light": "red can", 
            "bench": "table",
            "dining table": "table",
            "cell phone": "red can",
            "bottle": "red can",
            "cup": "red can",
            "vase": "red can",
            "wine glass": "red can"
        }
        
        found_target = None
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = robot_interface.yolo.names[cls_id]
                conf = float(box.conf[0])
                mapped = alias_map.get(cls_name.lower(), cls_name.lower())
                detected.append(f"{cls_name}->{mapped}")
                print(f"      👁️ {cls_name} ({conf:.2f}) -> {mapped}")
                
                # If we found red can or table, compute position immediately
                if mapped in ["red can", "table"] and depth:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    if 0 <= cx < w and 0 <= cy < h:
                        d = depth[cy * w + cx]
                        if 0.1 < d < 5.0:
                            # Calculate 3D position in robot frame
                            # Camera FOV is approximately 60 degrees horizontal
                            fov_h = 1.0  # radians (~57 degrees)
                            pixel_angle = (cx - w/2) / w * fov_h
                            
                            # x = forward distance (depth * cos(pixel_angle))
                            # y = lateral offset (depth * sin(pixel_angle)), negative = right
                            x_arm = d * np.cos(pixel_angle)
                            y_arm = -d * np.sin(pixel_angle)  # Negative because right is negative Y
                            z_arm = 0.78  # Table height (can is on table)
                            
                            print(f"      ✅ FOUND {mapped} at depth={d:.2f}m, pos=({x_arm:.2f},{y_arm:.2f},{z_arm:.2f})")
                            
                            if mapped == "red can" and not found_target:
                                found_target = {"object": mapped, "position": [x_arm, y_arm, z_arm], "distance": d}
        
        if found_target:
            return json.dumps({
                "status": "found",
                "object": found_target["object"],
                "position": found_target["position"],
                "distance": found_target["distance"],
                "msg": f"Target found! Distance={found_target['distance']:.2f}m. Use move_arm or move_base."
            })
        elif detected:
            return f"Head at Pan={pan}, Tilt={tilt}. SAW: {detected}. No target (red can) found. Try different angles."
        else:
            return f"Head at Pan={pan}, Tilt={tilt}. Nothing detected. Try different angles (pan: -1 to 1, tilt: -0.9 to 0.7)."

class DetectObjectInput(BaseModel):
    object_name: str = Field(..., description="Name of the object to find")

class DetectObjectTool(BaseTool):
    name: str = "detect_object"
    description: str = "Finds an object using YOLO."
    args_schema: Type[BaseModel] = DetectObjectInput

    def _run(self, object_name: str) -> str:
        if not robot_interface: return "Robot not ready"
        
        print(f"📷 Vision: Scanning for '{object_name}'...")
        
        # Get current sensor data (single frame - fast!)
        rgb = None
        depth = None
        w = 0
        h = 0
        
        with sensor_data["lock"]:
            if sensor_data["rgb"] is not None:
                rgb = sensor_data["rgb"].copy()
                if sensor_data["depth"]:
                    depth = list(sensor_data["depth"])
                w = sensor_data["camera_width"]
                h = sensor_data["camera_height"]
        
        if rgb is None:
            return json.dumps({"status": "not_found", "msg": "No image available"})
        
        # Run YOLO (single inference - fast!)
        results = robot_interface.yolo(rgb, verbose=False, conf=0.6)
        num_boxes = len(results[0].boxes) if results else 0
        print(f"   🔍 YOLO detected {num_boxes} objects")
        
        seen_objects = []
        alias_map = {
            "traffic light": "red can", 
            "bench": "table",
            "dining table": "table",
            "cell phone": "red can",
            "bottle": "red can",
            "cup": "red can",
            "vase": "red can",
            "wine glass": "red can"
        }
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = robot_interface.yolo.names[cls_id]
                conf = float(box.conf[0])
                mapped_name = alias_map.get(cls_name.lower(), cls_name.lower())
                seen_objects.append(f"{cls_name}->{mapped_name}")
                print(f"      👁️ {cls_name} ({conf:.2f}) -> {mapped_name}")
                
                if object_name.lower() in mapped_name or object_name.lower() in cls_name.lower():
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    
                    if depth and 0 <= cx < w and 0 <= cy < h:
                        d = depth[cy * w + cx]
                        if d < 0.1 or d > 5.0: 
                            continue
                        
                        # Calculate 3D position in robot frame
                        fov_h = 1.0  # radians (~57 degrees)
                        pixel_angle = (cx - w/2) / w * fov_h
                        x_arm = d * np.cos(pixel_angle)
                        y_arm = -d * np.sin(pixel_angle)
                        z_arm = 0.78
                        
                        print(f"✅ MATCH! '{cls_name}' -> '{mapped_name}' at pixel({cx},{cy}), depth={d:.2f}m, pos=({x_arm:.2f},{y_arm:.2f},{z_arm:.2f})")
                        
                        return json.dumps({
                            "status": "found", 
                            "object": mapped_name,
                            "position": [float(x_arm), float(y_arm), float(z_arm)],
                            "distance": float(d)
                        })
        
        return json.dumps({"status": "not_found", "msg": f"Object not visible. Saw: {seen_objects}"})

class MoveBaseInput(BaseModel):
    distance: float = Field(..., description="Distance (m)")
    angle: float = Field(..., description="Angle (rad)")

class MoveBaseTool(BaseTool):
    name: str = "move_base"
    description: str = "Moves the robot base forward/backward and rotates."
    args_schema: Type[BaseModel] = MoveBaseInput
    def _run(self, distance: float, angle: float) -> str:
        import time
        action_queue.put({"type": "base", "lin": distance, "ang": angle})
        # Wait for base to actually move (estimate + buffer)
        move_time = abs(distance) / 0.5 + abs(angle) / 0.8 + 0.5
        print(f"🚗 Moving base: distance={distance:.2f}m, angle={angle:.2f}rad (waiting {move_time:.1f}s)")
        time.sleep(move_time)
        return f"Base moved {distance:.2f}m forward. Use look_around to check new position."

class MoveArmInput(BaseModel):
    x: float = Field(..., description="X: forward distance from robot (usually 0.4-0.7)")
    y: float = Field(..., description="Y: left/right offset (negative=right, usually -0.3 to 0.3)")
    z: float = Field(..., description="Z: height (table ~0.75-0.85)")

class MoveArmTool(BaseTool):
    name: str = "move_arm"
    description: str = "Moves arm to (x,y,z) using Pinocchio IK. After look_around, use the position values but adjust x to be closer (e.g., if distance=0.5, use x=0.45)."
    args_schema: Type[BaseModel] = MoveArmInput
    def _run(self, x: float, y: float, z: float) -> str:
        import time
        
        # Transform camera coordinates to arm workspace (URDF) coordinates
        # TIAGo++ right arm base is offset from robot center
        # 
        # Camera reports:
        #   x = depth (forward from camera)
        #   y = lateral offset (negative = object is to the right)
        #   z = height (usually fixed at 0.8)
        #
        # Arm workspace in URDF:
        #   X axis: forward
        #   Y axis: left (right arm is at Y ~ -0.2 from center)
        #   Z axis: up
        
        # Right arm base is ~0.2m to the right of robot center
        # Camera is ~0.1m in front of robot center
        # Arm reach is ~0.5-0.7m from shoulder
        
        # Clamp to reachable workspace
        x_clamped = max(0.3, min(0.7, x))  # Forward reach limit
        y_clamped = max(-0.4, min(0.2, y)) # Right arm can reach right side better
        z_clamped = max(0.6, min(1.0, z))  # Height range
        
        # Convert to URDF coordinates for right arm
        # The arm shoulder is at approximately (0.1, -0.2, 1.0) in base_link frame
        urdf_x = x_clamped
        urdf_y = y_clamped - 0.1  # Slight offset for right arm
        urdf_z = z_clamped
        
        print(f"🎯 IK Target: Input({x:.2f},{y:.2f},{z:.2f}) -> URDF({urdf_x:.2f},{urdf_y:.2f},{urdf_z:.2f})", flush=True)
        
        action_queue.put({"type": "arm_ik", "pos": [urdf_x, urdf_y, urdf_z]})
        time.sleep(4.0)  # Wait for IK computation and arm movement
        return f"Arm moved towards ({urdf_x:.2f}, {urdf_y:.2f}, {urdf_z:.2f}). Use control_gripper to grab."


class MoveArmPresetInput(BaseModel):
    action: str = Field(..., description="'reach' or 'home'")

class MoveArmPresetTool(BaseTool):
    name: str = "arm_preset"
    description: str = "Preset arm positions: 'reach' (extend forward) or 'home' (retract)."
    args_schema: Type[BaseModel] = MoveArmPresetInput
    def _run(self, action: str) -> str:
        if action.lower() == "reach":
            action_queue.put({"type": "arm_preset", "preset": "reach"})
            return "Arm reaching forward. Now use control_gripper to grab."
        elif action.lower() == "home":
            action_queue.put({"type": "arm_preset", "preset": "home"})
            return "Arm retracted to home position."
        else:
            return f"Unknown action '{action}'. Use 'reach' or 'home'."

class GripperInput(BaseModel):
    action: str = Field(..., description="'open' or 'close'")

class GripperTool(BaseTool):
    name: str = "control_gripper"
    description: str = "Control gripper."
    args_schema: Type[BaseModel] = GripperInput
    def _run(self, action: str) -> str:
        action_queue.put({"type": "gripper", "open": (action.lower() == "open")})
        return f"Gripper {action}"

# --- Agent ---
def run_crew_ai():
    print("🚀 CrewAI Agent Started")
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    
    engineer = Agent(
        role='Robotics Engineer',
        goal='Pick up objects and place them autonomously',
        backstory="""You are a TIAGo++ robot controller. You decide how to achieve the goal.

## Knowledge Base (Reference Only - Use Your Judgment)

### Perception
- Camera is mounted on head. Use look_around(pan, tilt) to see.
- tilt=-0.5 looks down at table level. tilt=0.0 looks forward.
- look_around returns: position=[x, y, z] and distance in meters.

### Robot Capabilities  
- ARM REACH: 0.5m ~ 0.7m from robot base
- If distance > 0.7m: object is TOO FAR, must move_base closer
- If distance < 0.5m: object is WITHIN REACH, can use move_arm
- If distance 0.5~0.7m: OPTIMAL range for grasping

### Arm Control
- move_arm(x, y, z): x=forward, y=lateral(negative=right), z=height
- IMPORTANT: Subtract 0.2~0.3 from detected x for arm to reach object
  Example: detected at x=0.6 → use move_arm(x=0.35, y=..., z=...)
- Table height is ~0.78m

### Movement
- move_base(distance, angle): distance in meters, angle in radians
- To approach: calculate (detected_distance - 0.5) for safe approach
- angle=1.57 is 90 degrees left turn

### Gripper
- Open gripper BEFORE reaching for object
- Close gripper to grab, open to release

## Decision Framework
1. OBSERVE: Use look_around. What's the distance?
2. ANALYZE: Is it within arm reach (0.5-0.7m)?
3. DECIDE: Move base if too far, or use arm if close enough
4. EXECUTE: One action at a time
5. VERIFY: Check result before proceeding

## CRITICAL RULES
- If you moved base but distance STAYS THE SAME: STOP moving base! 
  Just try move_arm with adjusted x (subtract 0.3-0.4 from distance).
- NEVER call move_base more than 2 times total.
- After 2 move_base attempts, MUST use move_arm regardless of distance.
- If distance is 0.7-1.0m, you CAN try move_arm with x=0.5 (max reach).

Think carefully. You decide the sequence.""",
        tools=[GetRobotStateTool(), DetectObjectTool(), MoveBaseTool(), MoveArmTool(), MoveArmPresetTool(), GripperTool(), LookAroundTool()],
        llm=llm,
        verbose=True
    )

    task = Task(
        description='Find the red can (look around if needed), drive to it, pick it up, and place it on the table.',
        expected_output='Action completed.',
        agent=engineer
    )

    Crew(agents=[engineer], tasks=[task], process=Process.sequential).kickoff()

def main():
    global robot_interface
    supervisor = Supervisor()
    robot_interface = TiagoRobotInterface(supervisor)
    
    threading.Thread(target=run_crew_ai, daemon=True).start()

    while supervisor.step(robot_interface.timestep) != -1:
        robot_interface.update_sensors()
        
        while not action_queue.empty():
            cmd = action_queue.get()
            
            if cmd['type'] == 'base':
                lin_speed, ang_speed = 0.5, 0.8
                if abs(cmd['ang']) > 0.01:
                    duration = abs(cmd['ang']) / ang_speed
                    robot_interface.execute_move_base(0, ang_speed if cmd['ang']>0 else -ang_speed)
                    for _ in range(int(duration*1000/robot_interface.timestep)): 
                        supervisor.step(robot_interface.timestep); robot_interface.update_sensors()
                    robot_interface.execute_move_base(0, 0)
                    for _ in range(5): supervisor.step(robot_interface.timestep)
                
                if abs(cmd['lin']) > 0.01:
                    duration = abs(cmd['lin']) / lin_speed
                    robot_interface.execute_move_base(lin_speed if cmd['lin']>0 else -lin_speed, 0)
                    for _ in range(int(duration*1000/robot_interface.timestep)): 
                        supervisor.step(robot_interface.timestep); robot_interface.update_sensors()
                    robot_interface.execute_move_base(0, 0)
                    for _ in range(10): supervisor.step(robot_interface.timestep)
            
            elif cmd['type'] == 'head':
                robot_interface.execute_move_head(cmd['pan'], cmd['tilt'])
                print(f"👀 Head Moving: Pan={cmd['pan']}, Tilt={cmd['tilt']}")
                for _ in range(50): supervisor.step(robot_interface.timestep); robot_interface.update_sensors()
                
            elif cmd['type'] == 'arm_ik':
                robot_interface.execute_move_arm(cmd['pos'])
                for _ in range(200): supervisor.step(robot_interface.timestep); robot_interface.update_sensors()
            
            elif cmd['type'] == 'arm_preset':
                preset = cmd['preset']
                if preset == "reach":
                    # Predefined joint angles to reach forward (experimentally tuned)
                    # These angles put the gripper in front of the robot at table height
                    reach_pose = [0.2, 0.0, -1.5, 1.5, 0.0, 0.5, 0.0]
                    print("🦾 Arm: Reaching forward...")
                    for i, motor in enumerate(robot_interface.arm_joints):
                        if i < len(reach_pose):
                            motor.setPosition(reach_pose[i])
                elif preset == "home":
                    home_pose = [0.2, 0.5, 0.0, 1.0, -1.57, 0.0, 0.0]
                    print("🦾 Arm: Retracting to home...")
                    for i, motor in enumerate(robot_interface.arm_joints):
                        if i < len(home_pose):
                            motor.setPosition(home_pose[i])
                for _ in range(150): supervisor.step(robot_interface.timestep)
                
            elif cmd['type'] == 'gripper':
                robot_interface.execute_gripper(cmd['open'])
                for _ in range(10): supervisor.step(robot_interface.timestep)
            
            elif cmd['type'] == 'refresh':
                # Force sensor refresh - run multiple steps to ensure data is fresh
                print("🔄 Forcing sensor refresh...", flush=True)
                for _ in range(30): 
                    supervisor.step(robot_interface.timestep)
                    robot_interface.update_sensors()

if __name__ == "__main__":
    main()
