from controller import Supervisor
import sys

# === 이동 함수들 =====================================================

def move_stop(left_wheel, right_wheel):
    left_wheel.setPosition(0)
    right_wheel.setPosition(0)
    left_wheel.setVelocity(0)
    right_wheel.setVelocity(0)

def move_forward(left_wheel, right_wheel, speed=1.0):
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(float('inf'))
    left_wheel.setVelocity(speed)
    right_wheel.setVelocity(speed)

def move_backward(left_wheel, right_wheel, speed=1.0):
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(float('inf'))
    left_wheel.setVelocity(-speed)
    right_wheel.setVelocity(-speed)

def turn_left(left_wheel, right_wheel, speed=1.0):
    # 제자리 회전 (좌우 반대 속도)
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(float('inf'))
    left_wheel.setVelocity(-speed)
    right_wheel.setVelocity(speed)

def turn_right(left_wheel, right_wheel, speed=1.0):
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(float('inf'))
    left_wheel.setVelocity(speed)
    right_wheel.setVelocity(-speed)

# =====================================================================

# Supervisor 컨트롤러 생성 (Robot() 따로 만들지 말 것!)
supervisor = Supervisor()
timestep = int(supervisor.getBasicTimeStep())

print(f"기본 시간 스텝: {timestep}ms")

# (중요) 월드 파일(.wbt)에서 로봇 루트 노드에 붙인 DEF 이름을 넣어라.
# 예: DEF MY_ROBOT Pioneer3dx { ... }
ROBOT_DEF_NAME = "robot2"
robot_node = supervisor.getFromDef(ROBOT_DEF_NAME)
if robot_node is None:
    print(f"[에러] DEF {ROBOT_DEF_NAME} 로봇 노드를 찾을 수 없습니다. .wbt에서 DEF 이름을 확인하세요.")
    sys.exit(1)

# 장치들 가져오기 (Supervisor는 Robot을 상속하므로 getDevice 사용 가능)
left_wheel = supervisor.getDevice("MLW")
right_wheel = supervisor.getDevice("MRW")
distance_sensor = supervisor.getDevice("DS_0")
led = supervisor.getDevice("led")

# 초기 LED 끄기
if led:
    led.set(0x000000)

# 센서 enable
if distance_sensor:
    distance_sensor.enable(timestep)

step = 0

while supervisor.step(timestep) != -1:
    step += 1

    # 로봇 위치 출력 (10 스텝마다)
    if step % 10 == 0:
        pos = robot_node.getPosition()  # [x, y, z] in world coordinates
        print(f"Step {step}: 로봇 위치 = [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # 거리 센서 값 읽기
    distance_value = None
    if distance_sensor:
        distance_value = distance_sensor.getValue()
        if step % 10 == 0:
            print(f"거리 센서 값: {distance_value:.1f}mm")

    # 충돌 회피 로직
    if distance_value is not None and distance_value < 350:
        print(f"충돌 감지({distance_value:.1f} mm)! 회피 동작 실행")

        if led:
            led.set(0xff0000)
            print('LED ON')

        # 1) 잠깐 후진
        move_backward(left_wheel, right_wheel, 1.0)
        supervisor.step(timestep * 100)

        # 2) 왼쪽으로 회전
        turn_left(left_wheel, right_wheel, 1.0)
        supervisor.step(timestep * 100)

        print("회피 동작 완료")
    else:
        if led:
            led.set(0x000000)
            # LED OFF 메시지 너무 많이 찍히면 로그 폭주하므로 step % 10 에만 찍어도 됨
            if step % 10 == 0:
                print('LED OFF')

        # 정상 전진
        move_forward(left_wheel, right_wheel, 1.0)
