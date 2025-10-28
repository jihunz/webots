from controller import Supervisor
import math


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
    left_wheel.setPosition(-float('inf'))
    right_wheel.setPosition(-float('inf'))
    left_wheel.setVelocity(-speed)
    right_wheel.setVelocity(-speed)


def move_left(left_wheel, right_wheel, speed=1.0):
    left_wheel.setPosition(-float('inf'))
    right_wheel.setPosition(float('inf'))
    left_wheel.setVelocity(-speed)
    right_wheel.setVelocity(speed)


def move_right(left_wheel, right_wheel, speed=1.0):
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(-float('inf'))
    left_wheel.setVelocity(speed)
    right_wheel.setVelocity(-speed)


def html_format(message):
    message = message.replace("<", "&lt;")
    message = message.replace(">", "&gt;")
    message = message.replace("\n", "<br>")
    return message


robot = Supervisor()
timestep = int(robot.getBasicTimeStep())

print(f"기본 시간 스텝: {timestep}ms")

# 로봇 노드 가져오기
robot_node = robot.getSelf()

# 장치들 가져오기
left_wheel = robot.getDevice("MLW")
right_wheel = robot.getDevice("MRW")
distance_sensor = robot.getDevice("DS_0")
led = robot.getDevice("led")
led.set(0)

# 장치 활성화
if distance_sensor:
    distance_sensor.enable(timestep)

step = 0
while robot.step(timestep) != -1:
    step += 1

    # 로봇 위치 출력 (10스텝마다)
    if step % 10 == 0:
        position = robot_node.getPosition()
        print(f"Step {step}: 로봇 위치 = [{position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}]")

        x, y, z = position
        position_string = f"x: {x:.2f}, y: {y:.2f}, z: {z:.2f}"

        rotation_field = robot_node.getField("rotation")
        rotation_values = rotation_field.getSFRotation()
        # rotation_values is [x, y, z, angle] where x,y,z is axis and angle is in radians
        axis_x, axis_y, axis_z, angle_rad = rotation_values
        angle_deg = angle_rad * 180 / math.pi
        orientation_string = f"Angle [deg]: {angle_deg:.2f}, axis: [{axis_x:.2f}, {axis_y:.2f}, {axis_z:.2f}]"
        reply = f"Position [m] = {position_string}\nOrientation = {orientation_string}\nMessage = {message}"
        reply = html_format(reply)

        robot.wwiSendText(reply)

    # 거리 센서 값 확인
    if distance_sensor:
        distance_value = distance_sensor.getValue()
        if step % 10 == 0:
            print(f"거리 센서 값: {distance_value:.1f}mm")

    # 충돌 회피 로직
    if distance_sensor.getValue() < 350:
        print(f"충돌 감지({distance_sensor.getValue():.1f} mm)!  회피 동작 실행")

        led.set(255)
        print('LED ON')

        move_backward(left_wheel, right_wheel, 1.0)
        robot.step(timestep * 100)

        move_left(left_wheel, right_wheel, 1.0)
        robot.step(timestep * 100)

        print("회피 동작 완료")

        led.set(0)
        print('LED OFF')
        move_forward(left_wheel, right_wheel, 1.0)

    message = robot.wwiReceiveText()
    if message:
        # Print the message if not None
        print('USER_MESSAGE: ' + message)
        try:
            cmd, speed, duration = message.split(" ")
            speed = float(speed)
            duration = float(duration)  # seconds
        except:
            cmd = message.split(" ")[0]
            speed = 1.0
            duration = 1.0

        if cmd == "forward":
            move_forward(left_wheel, right_wheel, speed)
        elif cmd == "backward":
            move_backward(left_wheel, right_wheel, speed)
        elif cmd == "left":
            move_left(left_wheel, right_wheel, speed)
        elif cmd == "right":
            move_right(left_wheel, right_wheel, speed)
        elif cmd == "stop":
            move_stop(left_wheel, right_wheel)