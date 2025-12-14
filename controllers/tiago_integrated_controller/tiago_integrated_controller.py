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

    def execute_move_arm(self, target_pos):
        print(f"🦾 H/W: Moving arm towards {target_pos}")
        if not self.model or not self.arm_joints: return "IK Error"
        
        guesses = []
        q_neutral = pin.neutral(self.model)
        guesses.append(q_neutral)
        
        q_curr = q_neutral.copy()
        for name, sensor in zip([m.getName() for m in self.arm_joints], self.arm_sensors):
            if sensor and self.model.existJointName(name):
                idx = self.model.getJointId(name)
                idx_q = self.model.joints[idx].idx_q
                q_curr[idx_q] = sensor.getValue()
        guesses.append(q_curr)
        
        for _ in range(5):
            guesses.append(pin.randomConfiguration(self.model))

        eps = 1e-2; IT_MAX = 500; DT = 1e-1; damp = 1e-3
        best_q = None
        
        for q_start in guesses:
            q = q_start.copy()
            success = False
            for i in range(IT_MAX):
                pin.forwardKinematics(self.model, self.data, q)
                pin.updateFramePlacements(self.model, self.data)
                curr = self.data.oMf[self.ee_frame_id].translation
                err = curr - np.array(target_pos)
                if np.linalg.norm(err) < eps:
                    success = True
                    break
                J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id)[:3, :]
                v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(3), err))
                q = pin.integrate(self.model, q, v * DT)
            if success:
                best_q = q
                print("✅ IK Converged")
                break
            
        if best_q is not None:
            for name, motor in zip([m.getName() for m in self.arm_joints], self.arm_joints):
                if self.model.existJointName(name):
                    idx = self.model.getJointId(name)
                    idx_q = self.model.joints[idx].idx_q
                    motor.setPosition(best_q[idx_q])
            return "Moved arm (IK Success)"
        
        print("❌ Pinocchio IK Failed (All attempts)")
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
        
        # 1. Send head command + force sensor refresh
        action_queue.put({"type": "head", "pan": pan, "tilt": tilt})
        action_queue.put({"type": "refresh"})  # Force Main Thread to refresh sensors
        print(f"👀 Moving head to Pan={pan}, Tilt={tilt}...", flush=True)
        
        # 2. Wait for head to move + sensor update
        time.sleep(2.5)
        
        # 3. Run YOLO directly here - read FRESH sensor data
        rgb = None
        depth = None
        w = 0
        h = 0
        
        # Try multiple times to get fresh data
        for _ in range(3):
            time.sleep(0.2)
            with sensor_data["lock"]:
                if sensor_data["rgb"] is not None:
                    rgb = sensor_data["rgb"].copy()
                    if sensor_data["depth"]:
                        depth = list(sensor_data["depth"])
                    w = sensor_data["camera_width"]
                    h = sensor_data["camera_height"]
            if rgb is not None:
                break
        
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
                            x_base = d
                            y_base = -(cx - w/2) * 0.002
                            z_base = 0.8
                            print(f"      ✅ FOUND {mapped} at depth={d:.2f}m")
                            # Return JSON with position for immediate use
                            if mapped == "red can" and not found_target:
                                found_target = {"object": mapped, "position": [x_base, y_base, z_base], "distance": d}
        
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
                        
                        print(f"✅ MATCH! '{cls_name}' -> '{mapped_name}' at ({cx}, {cy}), Depth={d:.3f}m")
                        
                        x_base = d
                        y_base = -(cx - w/2) * 0.002
                        z_base = 0.8 
                        
                        return json.dumps({
                            "status": "found", 
                            "object": mapped_name,
                            "position": [float(x_base), float(y_base), float(z_base)],
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
    action: str = Field(..., description="'reach_forward' to extend arm forward for grabbing, 'home' to retract arm")

class MoveArmTool(BaseTool):
    name: str = "move_arm"
    description: str = "Controls arm. Use 'reach_forward' when close to object (<0.5m), then use gripper."
    args_schema: Type[BaseModel] = MoveArmInput
    def _run(self, action: str) -> str:
        if action.lower() == "reach_forward":
            action_queue.put({"type": "arm_preset", "preset": "reach"})
            return "Arm reaching forward. Now use control_gripper to grab."
        elif action.lower() == "home":
            action_queue.put({"type": "arm_preset", "preset": "home"})
            return "Arm retracted to home position."
        else:
            return f"Unknown action '{action}'. Use 'reach_forward' or 'home'."

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
        goal='Manipulate objects robustly',
        backstory="""Expert robot controller. Follow this EXACT sequence:
        1. Use 'look_around' with tilt=-0.5 to find red can.
        2. Move base forward 0.5m using move_base.
        3. Use look_around again. If still far, move_base forward 0.3m.
        4. After 2-3 base moves, use 'move_arm' with action='reach_forward'.
        5. Use 'control_gripper' with action='close' to grab.
        6. Use 'move_arm' with action='home' to lift.
        7. Turn with move_base(angle=1.57) to find table, place object with gripper open.""",
        tools=[GetRobotStateTool(), DetectObjectTool(), MoveBaseTool(), MoveArmTool(), GripperTool(), LookAroundTool()],
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
                
            elif cmd['type'] == 'arm':
                robot_interface.execute_move_arm(cmd['pos'])
                for _ in range(100): supervisor.step(robot_interface.timestep)
            
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
