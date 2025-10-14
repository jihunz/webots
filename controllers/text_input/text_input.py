# http://localhost:1234/robot_windows/text_input/text_input.html?name=e-puck

PATH_WEBOTS_CONTROLLER = "C:/Program Files/Webots/lib/controller/python"

import sys, math

sys.path.append(PATH_WEBOTS_CONTROLLER)
from controller import Supervisor


def html_format(message):
    message = message.replace("<", "&lt;")
    message = message.replace(">", "&gt;")
    message = message.replace("\n", "<br>")
    return message


def move_stop(left_motor, right_motor):
    print("move_stop")

    right_motor.setPosition(0)
    left_motor.setPosition(0)
    left_motor.setVelocity(0)
    right_motor.setVelocity(0)


def move_forward(robot, left_motor, right_motor, speed, duration):
    print("move_forward", speed, duration)
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(speed)
    right_motor.setVelocity(speed)

    # move for the specified time
    start_time = robot.getTime()
    while robot.getTime() - start_time < duration:
        robot.step(int(robot.getBasicTimeStep()))

    move_stop(left_motor, right_motor)


def move_backward(robot, left_motor, right_motor, speed, duration):
    print("move_backward", speed, duration)
    left_motor.setPosition(-float('inf'))
    right_motor.setPosition(-float('inf'))
    left_motor.setVelocity(-speed)
    right_motor.setVelocity(-speed)

    start_time = robot.getTime()
    while robot.getTime() - start_time < duration:
        robot.step(int(robot.getBasicTimeStep()))

    move_stop(left_motor, right_motor)


def move_left(robot, left_motor, right_motor, speed, duration):
    print("move_left", speed, duration)
    left_motor.setPosition(-float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(-speed)
    right_motor.setVelocity(speed)

    start_time = robot.getTime()
    while robot.getTime() - start_time < duration:
        robot.step(int(robot.getBasicTimeStep()))

    move_stop(left_motor, right_motor)


def move_right(robot, left_motor, right_motor, speed, duration):
    print("move_right", speed, duration)
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(-float('inf'))
    left_motor.setVelocity(speed)
    right_motor.setVelocity(-speed)

    start_time = robot.getTime()
    while robot.getTime() - start_time < duration:
        robot.step(int(robot.getBasicTimeStep()))

    move_stop(left_motor, right_motor)


t = 0
robot = Supervisor()
timestep = int(robot.getBasicTimeStep())
left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

while robot.step(timestep) != -1:
    # Receive a message from the robot window
    message = robot.wwiReceiveText()
    if message:
        # Print the message if not None
        print('USER_MESSAGE: ' + message)
        try:
            cmd, speed, duration = message.split(" ")
            speed = float(speed)
            duration = float(duration)  # seconds
        except:
            cmd = message
            speed = 100.0
            duration = 5.0

        if cmd == "forward":
            move_forward(robot, left_motor, right_motor, speed, duration)
        elif cmd == "backward":
            move_backward(robot, left_motor, right_motor, speed, duration)
        elif cmd == "left":
            move_left(robot, left_motor, right_motor, speed, duration)
        elif cmd == "right":
            move_right(robot, left_motor, right_motor, speed, duration)
        else:
            move_stop(left_motor, right_motor)

    # Get the robot's position
    robot_node = robot.getSelf()
    position = robot_node.getPosition()
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

