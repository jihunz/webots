from controller import Supervisor, Keyboard

# --- 초기화 ---
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

# 키보드 활성화
keyboard = Keyboard()
keyboard.enable(timestep)

# 모터 초기화
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

left_motor.setPosition(float('inf'))
right_motor.setPosition(float('inf'))

speed = 0.0
max_speed = 100.0
speed_step = 20.0

print("Keyboard control enabled: W=속도증가, ↑↓←→=방향")

# --- 메인 루프 ---
while robot.step(timestep) != -1:
    key = keyboard.getKey()

    # 속도 제어
    if key == ord('W'):
        speed = min(speed + speed_step, max_speed)
        print(f"Speed up → {speed:.1f}")

    # 방향키 입력 (↑, ↓, ←, →)
    if key == Keyboard.UP:
        left_motor.setVelocity(speed)
        right_motor.setVelocity(speed)
    elif key == Keyboard.DOWN:
        left_motor.setVelocity(-speed)
        right_motor.setVelocity(-speed)
    elif key == Keyboard.LEFT:
        left_motor.setVelocity(-speed)
        right_motor.setVelocity(speed)
    elif key == Keyboard.RIGHT:
        left_motor.setVelocity(speed)
        right_motor.setVelocity(-speed)
    else:
        # 입력 없으면 멈춤
        left_motor.setVelocity(0)
        right_motor.setVelocity(0)