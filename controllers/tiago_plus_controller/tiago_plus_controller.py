from controller import Robot, Camera, Motor, DistanceSensor, PositionSensor, Keyboard
import math
import numpy as np

# TIAGo++ Constants
MAX_WHEEL_SPEED = 1.0  # rad/s
ARM_Joints = 7  # 7-DOF arm
GRIPPER_MAX_OPEN = 0.09 # m

class TiagoController:
    def __init__(self):
        self.robot = Robot()
        self.timestep = int(self.robot.getBasicTimeStep())
        
        # ------------------------------------------------------------------
        # 1. Base (Wheels) Initialization
        # ------------------------------------------------------------------
        self.wheels = []
        self.wheel_names = ['wheel_left_joint', 'wheel_right_joint']
        for name in self.wheel_names:
            motor = self.robot.getDevice(name)
            if motor:
                motor.setPosition(float('inf'))
                motor.setVelocity(0.0)
                self.wheels.append(motor)
            else:
                print(f"Warning: Motor {name} not found")
            
        # ------------------------------------------------------------------
        # 2. Head Initialization
        # ------------------------------------------------------------------
        self.head_motors = {
            'pan': self.robot.getDevice('head_1_joint'),
            'tilt': self.robot.getDevice('head_2_joint')
        }
        
        # ------------------------------------------------------------------
        # 3. Arms Initialization (Left & Right)
        # ------------------------------------------------------------------
        self.arms = {'left': [], 'right': []}
        # TIAGo arm joint names: arm_{side}_{1-7}_joint
        for side in ['left', 'right']:
            for i in range(1, 8):
                name = f'arm_{side}_{i}_joint'
                motor = self.robot.getDevice(name)
                if motor:
                    # Position sensor for feedback
                    sensor = self.robot.getDevice(f'{name}_sensor')
                    if sensor:
                        sensor.enable(self.timestep)
                    self.arms[side].append(motor)
                else:
                    print(f"Warning: Arm motor {name} not found")

        # ------------------------------------------------------------------
        # 4. Grippers Initialization
        # ------------------------------------------------------------------
        # TIAGo gripper usually has 'gripper_{side}_left_finger_joint' and 'right'
        self.grippers = {'left': {}, 'right': {}}
        for side in ['left', 'right']:
            try:
                self.grippers[side]['left'] = self.robot.getDevice(f'gripper_{side}_left_finger_joint')
                self.grippers[side]['right'] = self.robot.getDevice(f'gripper_{side}_right_finger_joint')
            except Exception:
                print(f"Warning: Gripper for {side} not found")

        # ------------------------------------------------------------------
        # 5. Sensors (Camera & Lidar)
        # ------------------------------------------------------------------
        # RGB-D Camera (Head)
        # Note: Check actual device names in your wbt file
        self.camera_rgb = self.robot.getDevice('xtion_rgb') 
        self.camera_depth = self.robot.getDevice('xtion_depth')
        
        if self.camera_rgb:
            self.camera_rgb.enable(self.timestep)
        if self.camera_depth:
            self.camera_depth.enable(self.timestep)
            
        # Lidar (Base)
        self.lidar = self.robot.getDevice('Hokuyo URG-04LX-UG01')
        if self.lidar:
            self.lidar.enable(self.timestep)
            self.lidar.enablePointCloud()

    # ==================================================================
    # Core Methods
    # ==================================================================

    def step(self):
        """Perform one simulation step"""
        return self.robot.step(self.timestep) != -1

    def delay(self, seconds):
        """Wait for specific duration"""
        steps = int(seconds * 1000 / self.timestep)
        for _ in range(steps):
            if not self.step():
                break

    # --- Head Control ---
    def look_at(self, pan, tilt):
        """
        Move head to look at specific direction
        pan: horizontal angle (rad)
        tilt: vertical angle (rad)
        """
        if self.head_motors['pan']: self.head_motors['pan'].setPosition(pan)
        if self.head_motors['tilt']: self.head_motors['tilt'].setPosition(tilt)

    # --- Base Control ---
    def set_wheel_velocity(self, left_vel, right_vel):
        """Direct velocity control for differential drive"""
        if len(self.wheels) >= 2:
            self.wheels[0].setVelocity(left_vel)
            self.wheels[1].setVelocity(right_vel)

    def move_base(self, linear_x, angular_z):
        """
        Cmd_vel style base control
        linear_x: forward speed (m/s)
        angular_z: rotation speed (rad/s)
        """
        # Simple kinematics for differential drive
        # Assuming wheel radius and distance (standard TIAGo values)
        WHEEL_RADIUS = 0.0985
        WHEEL_SEPARATION = 0.4044
        
        left_vel = (linear_x - angular_z * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        right_vel = (linear_x + angular_z * WHEEL_SEPARATION / 2) / WHEEL_RADIUS
        
        # Clamp to max speed
        left_vel = max(min(left_vel, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)
        right_vel = max(min(right_vel, MAX_WHEEL_SPEED), -MAX_WHEEL_SPEED)
        
        self.set_wheel_velocity(left_vel, right_vel)

    def stop_base(self):
        self.set_wheel_velocity(0.0, 0.0)

    # --- Arm Control ---
    def set_arm_joints(self, side, target_positions):
        """
        Set target positions for 7 joints of an arm
        side: 'left' or 'right'
        target_positions: list of 7 floats (radians)
        """
        if len(target_positions) != 7:
            print(f"Error: 7 joint angles required, got {len(target_positions)}")
            return
            
        motors = self.arms.get(side)
        if not motors:
            print(f"Error: Invalid arm side '{side}'")
            return
            
        for motor, pos in zip(motors, target_positions):
            if motor:
                motor.setPosition(pos)

    def get_arm_joints(self, side):
        """Get current joint positions"""
        # Note: Needs PositionSensors to be enabled in init
        # This is a placeholder as motors.getTargetPosition is simpler for now
        # Ideally use: sensor.getValue()
        pass

    def tuck_arms(self):
        """Move arms to safe 'home' configuration"""
        # Standard safe pose for TIAGo
        safe_pose_left = [0.2, -1.34, -0.2, 1.94, -1.57, 1.37, 0.0]
        safe_pose_right = [0.2, -1.34, -0.2, 1.94, -1.57, 1.37, 0.0]
        
        self.set_arm_joints('left', safe_pose_left)
        self.set_arm_joints('right', safe_pose_right)

    # --- Gripper Control ---
    def control_gripper(self, side, open_width):
        """
        Control gripper width
        side: 'left' or 'right'
        open_width: 0.0 (closed) to 1.0 (fully open)
        """
        # TIAGo gripper is parallel. 0.045m per finger approx for max open
        # Value is usually in meters for prismatic joint or rad for revolute
        # Assuming Webots model uses prismatic joints for parallel gripper
        
        target = min(max(open_width, 0.0), 1.0) * 0.045 # 4.5cm per finger
        
        if self.grippers[side].get('left'):
            self.grippers[side]['left'].setPosition(target)
        if self.grippers[side].get('right'):
            self.grippers[side]['right'].setPosition(target)

    def open_gripper(self, side):
        self.control_gripper(side, 1.0)

    def close_gripper(self, side):
        self.control_gripper(side, 0.0)

    # --- Sensor ---
    def get_camera_image(self):
        """Returns raw RGB image if camera enabled"""
        if self.camera_rgb:
            return self.camera_rgb.getImage()
        return None

# ==================================================================
# Usage Example
# ==================================================================
if __name__ == "__main__":
    controller = TiagoController()
    
    # 1. 초기화 대기
    print("Initializing TIAGo++...")
    controller.delay(1.0)
    
    # 2. Look: 주변 스캔
    print("Looking around...")
    controller.look_at(0.0, -0.5) # 아래를 봄
    controller.delay(2.0)
    
    # 3. Approach: 앞으로 이동
    print("Approaching...")
    controller.move_base(0.3, 0.0) # 0.3 m/s 전진
    controller.delay(3.0)
    controller.stop_base()
    
    # 4. Plan/Execute: 팔 뻗기 (왼팔)
    print("Reaching with left arm...")
    # 예시: 앞으로 뻗는 자세 (IK Solver가 계산해야 할 값)
    reach_pose = [1.6, 0.3, 0.0, 1.5, 1.57, 0.0, 0.0] 
    controller.set_arm_joints('left', reach_pose)
    controller.delay(3.0)
    
    # 5. Grasp: 잡기
    print("Closing gripper...")
    controller.open_gripper('left') # 먼저 열고
    controller.delay(1.0)
    controller.close_gripper('left') # 닫기
    controller.delay(1.0)
    
    # 6. Retreat: 복귀
    print("Retreating...")
    controller.tuck_arms()
    controller.move_base(-0.2, 0.0) # 후진
    controller.delay(2.0)
    controller.stop_base()

