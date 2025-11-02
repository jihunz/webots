from controller import Robot
import time

robot = Robot()
timestep = int(robot.getBasicTimeStep())

GRIPPER_MOTORS = [
    "finger_1_joint_1",
    "finger_2_joint_1",
    "finger_middle_joint_1"
]

motors = {}

for name in GRIPPER_MOTORS:
    m = robot.getDevice(name)
    motors[name] = m
    m.setPosition(float('inf'))     # 🔥 position 제어 끄고 velocity 모드 전환
    m.setVelocity(0.0)

def open_gripper():
    print("👐 Open gripper")
    for m in motors.values():
        m.setVelocity(-0.5)         # 음수 → 벌리기
    for _ in range(50):             # 약 1.5초 정도
        if robot.step(timestep) == -1:
            break
    for m in motors.values():
        m.setVelocity(0.0)

def close_gripper():
    print("✊ Close gripper")
    for m in motors.values():
        m.setVelocity(0.5)          # 양수 → 닫기
    for _ in range(50):
        if robot.step(timestep) == -1:
            break
    for m in motors.values():
        m.setVelocity(0.0)

print("🏁 Start gripper velocity test")
open_gripper()
time.sleep(1)
close_gripper()
time.sleep(1)
open_gripper()
print("✅ Test done")
