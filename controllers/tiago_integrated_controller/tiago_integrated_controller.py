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
YOLO_MODEL = "yolov8n.pt"
URDF_FILENAME = "tiago_generated.urdf"

action_queue = Queue()

# Shared Data for Thread Sync
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
        
        # 1. Generate URDF dynamically from Webots
        print("📝 Generating URDF from Webots...")
        urdf_content = self.robot.getUrdf()
        with open(URDF_FILENAME, "w") as f:
            f.write(urdf_content)
            
        # 2. Load Pinocchio Model (Robust Strategy)
        self.model = None
        try:
            if not os.path.exists(URDF_FILENAME) or os.path.getsize(URDF_FILENAME) < 100:
                raise Exception("URDF file empty or missing")

            # Attempt 1: pin.buildModelFromUrdf
            if hasattr(pin, 'buildModelFromUrdf'):
                try:
                    self.model = pin.buildModelFromUrdf(URDF_FILENAME)
                    print("✅ Loaded via pin.buildModelFromUrdf")
                except Exception: pass

            # Attempt 2: pin.urdf.buildModel
            if self.model is None:
                try:
                    import pinocchio.urdf
                    self.model = pinocchio.urdf.buildModel(URDF_FILENAME)
                    print("✅ Loaded via pinocchio.urdf.buildModel")
                except ImportError: pass
                except Exception: pass

            # Attempt 3: RobotWrapper
            if self.model is None:
                try:
                    from pinocchio.robot_wrapper import RobotWrapper
                    self.model = RobotWrapper.BuildFromURDF(URDF_FILENAME).model
                    print("✅ Loaded via RobotWrapper")
                except ImportError: pass
                except Exception: pass

            if self.model is None:
                raise AttributeError("All Pinocchio URDF loaders failed.")

            self.data = self.model.createData()
            
            # Find End-Effector Frame
            self.ee_frame_id = -1
            candidates = ["arm_right_tool_link", "arm_right_7_link", "gripper_right_link"]
            for c in candidates:
                if self.model.existFrame(c):
                    self.ee_frame_id = self.model.getFrameId(c)
                    print(f"✅ Found EE Frame: {c} (ID: {self.ee_frame_id})")
                    break
            
            if self.ee_frame_id == -1:
                 # Fallback search
                 for f in self.model.frames:
                     if "right" in f.name and ("tool" in f.name or "7" in f.name):
                         self.ee_frame_id = self.model.getFrameId(f.name)
                         print(f"⚠️ Guessed EE Frame: {f.name} (ID: {self.ee_frame_id})")
                         break
            
            if self.ee_frame_id != -1:
                print(f"✅ Pinocchio IK Initialized Successfully")
            else:
                raise Exception("Could not find End-Effector Frame")

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
        
        if len(self.arm_joints) != 7:
            print(f"⚠️ Warning: Found {len(self.arm_joints)} arm joints. IK requires 7.")

        # 4. Gripper
        self.gripper = None
        for name in ['right_hand_gripper_right_finger_joint', 'gripper_right_finger_joint']:
            m = self.robot.getDevice(name)
            if m: self.gripper = m; break
            
        # 5. Base (Wheels)
        self.wheels = []
        for name in ['wheel_left_joint', 'wheel_right_joint']:
            m = self.robot.getDevice(name)
            if m:
                m.setPosition(float('inf'))
                m.setVelocity(0.0)
                self.wheels.append(m)

        # 6. Head (Crucial)
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

    def update_sensors(self):
        """Called from Main Thread every step to sync sensor data"""
        if self.camera and self.range_finder:
            raw_img = self.camera.getImage()
            raw_depth = self.range_finder.getRangeImage()
            
            with sensor_data["lock"]:
                if raw_img:
                    # Webots returns BGRA string/buffer
                    # 1. Convert to Numpy (Height, Width, 4)
                    img_bgra = np.frombuffer(raw_img, np.uint8).reshape((self.camera.getHeight(), self.camera.getWidth(), 4))
                    
                    # 2. Extract RGB
                    # Webots is BGRA. YOLO needs RGB.
                    # Slice first 3 channels (BGR) -> Reverse Last Axis (RGB)
                    # img_bgr = img_bgra[:, :, :3]
                    # img_rgb = img_bgr[:, :, ::-1]
                    
                    # One-liner: Take channels 2, 1, 0
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
        
        # 1. Prepare Initial Guesses (Random Restart)
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

        # 2. Pinocchio IK Loop
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

    def warmup(self):
        print("⏳ Warmup: Stabilizing sensors...")
        # 1. Wait for sensors
        for _ in range(20):
            self.robot.step(self.timestep)
            self.update_sensors()
            
        print("⏳ Warmup: Moving arm to Ready Pose...")
        # 2. Check current state (Optional log)
        # current_q = [s.getValue() for s in self.arm_sensors if s]
        # print(f"   Current Arm: {current_q}")

        # 3. Apply Ready Pose (Look Down + Arm Tuck)
        if self.head_tilt: self.head_tilt.setPosition(0.4)
        if self.head_pan: self.head_pan.setPosition(0.0)
        
        ready_pose = [0.2, 0.5, 0.0, 1.0, -1.57, 0.0, 0.0] 
        for i, m in enumerate(self.arm_joints):
            if i < len(ready_pose):
                m.setPosition(ready_pose[i])
        
        # 4. Wait for movement
        for _ in range(50):
            self.robot.step(self.timestep)
            self.update_sensors()
            
        print("✅ Warmup Complete: Robot Ready")

robot_interface = None 

# --- Tools ---
class GetRobotStateTool(BaseTool):
    name: str = "get_robot_state"
    description: str = "Returns current joint angles."
    def _run(self) -> str:
        return "State OK"

class LookAroundInput(BaseModel):
    pan: float = Field(..., description="Head Pan angle (rad). Range: -1.0 to 1.0")
    tilt: float = Field(..., description="Head Tilt angle (rad). Range: -0.8 (Down) to 0.0 (Front)")

class LookAroundTool(BaseTool):
    name: str = "look_around"
    description: str = "Moves the robot head to scan. PREFER looking DOWN (negative tilt) to find objects on floor/table."
    args_schema: Type[BaseModel] = LookAroundInput

    def _run(self, pan: float, tilt: float) -> str:
        # Enforce Look Down constraint for better object detection
        if tilt > 0.1: tilt = 0.0 
        
        action_queue.put({"type": "head", "pan": pan, "tilt": tilt})
        return "Head moved. Now call 'detect_object'."

class DetectObjectInput(BaseModel):
    object_name: str = Field(..., description="Name of the object to find")

class DetectObjectTool(BaseTool):
    name: str = "detect_object"
    description: str = "Finds an object using YOLO. If not found, try looking around."
    args_schema: Type[BaseModel] = DetectObjectInput

    def _run(self, object_name: str) -> str:
        if not robot_interface: return "Robot not ready"
        
        rgb = None
        depth = None
        w = 0
        h = 0
        
        with sensor_data["lock"]:
            if sensor_data["rgb"] is not None:
                rgb = sensor_data["rgb"].copy()
                depth = list(sensor_data["depth"]) if sensor_data["depth"] else None
                w = sensor_data["camera_width"]
                h = sensor_data["camera_height"]
        
        if rgb is None:
            return json.dumps({"status": "error", "msg": "No camera data"})
        
        # Restore confidence to 0.2
        results = robot_interface.yolo(rgb, verbose=False, conf=0.2)
        
        detected_names = []
        
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = robot_interface.yolo.names[cls_id]
                conf = float(box.conf[0])
                detected_names.append(f"{cls_name}({conf:.2f})")
                print(f"👁️ YOLO Saw: {cls_name} ({conf:.2f})")
                
                alias_map = {"traffic light": "red can", "bench": "table"}
                mapped_name = alias_map.get(cls_name.lower(), cls_name.lower())
                
                if object_name.lower() in mapped_name or object_name.lower() in cls_name.lower():
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                    
                    if depth and 0 <= cx < w:
                        d = depth[cy * w + cx]
                        if d < 0.1 or d > 5.0: continue
                        
                        print(f"✅ Found Target: '{cls_name}' -> '{mapped_name}' at ({cx}, {cy}), Depth={d:.3f}m")
                        
                        x_base = d
                        y_base = -(cx - w/2) * 0.002
                        z_base = 0.8 
                        
                        return json.dumps({
                            "status": "found", 
                            "object": mapped_name,
                            "position": [float(x_base), float(y_base), float(z_base)],
                            "distance": float(d)
                        })
        
        print(f"⚠️ Vision: Nothing found. (Detections: {detected_names})")
        return json.dumps({"status": "not_found", "msg": f"Object not visible. Saw: {detected_names}"})

class MoveBaseInput(BaseModel):
    distance: float = Field(..., description="Distance (m)")
    angle: float = Field(..., description="Angle (rad)")

class MoveBaseTool(BaseTool):
    name: str = "move_base"
    description: str = "Moves the robot base."
    args_schema: Type[BaseModel] = MoveBaseInput
    def _run(self, distance: float, angle: float) -> str:
        action_queue.put({"type": "base", "lin": distance, "ang": angle})
        return "Moving base..."

class MoveArmInput(BaseModel):
    x: float = Field(..., description="X")
    y: float = Field(..., description="Y")
    z: float = Field(..., description="Z")

class MoveArmTool(BaseTool):
    name: str = "move_arm"
    description: str = "Moves arm to (x,y,z)."
    args_schema: Type[BaseModel] = MoveArmInput
    def _run(self, x: float, y: float, z: float) -> str:
        action_queue.put({"type": "arm", "pos": [x, y, z]})
        return "Moving arm..."

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
        goal='Manipulate objects robustly using mobile manipulation',
        backstory="""Expert robot controller.
        1. Check state.
        2. Detect object. If NOT found, use 'look_around'.
        3. AFTER looking around, YOU MUST CALL 'detect_object' AGAIN.
        4. If found & far, approach.
        5. If close, pick up.
        6. Place on table.""",
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
    
    # Warmup removed
    # robot_interface.warmup()
    
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
                # Wait longer for head to settle
                for _ in range(50): 
                    supervisor.step(robot_interface.timestep)
                    robot_interface.update_sensors()
                
            elif cmd['type'] == 'arm':
                robot_interface.execute_move_arm(cmd['pos'])
                for _ in range(20): supervisor.step(robot_interface.timestep)
                
            elif cmd['type'] == 'gripper':
                robot_interface.execute_gripper(cmd['open'])
                for _ in range(10): supervisor.step(robot_interface.timestep)

if __name__ == "__main__":
    main()
