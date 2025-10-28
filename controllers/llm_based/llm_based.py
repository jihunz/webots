from controller import Supervisor, Robot
from openai import OpenAI
import os
import time
import dotenv
import json
import threading
from queue import Queue


# ---------------- 이동 제어 함수들 ----------------

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

def move_left(left_wheel, right_wheel, speed=1.0):
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(float('inf'))
    left_wheel.setVelocity(-speed)
    right_wheel.setVelocity(speed)

def move_right(left_wheel, right_wheel, speed=1.0):
    left_wheel.setPosition(float('inf'))
    right_wheel.setPosition(float('inf'))
    left_wheel.setVelocity(speed)
    right_wheel.setVelocity(-speed)


# ---------------- 유틸 ----------------

def html_format(message: str) -> str:
    msg = message.replace("<", "&lt;")
    msg = msg.replace(">", "&gt;")
    msg = msg.replace("\n", "<br>")
    return msg


# ---------------- 명령 큐 / 실행 스레드 ----------------

command_queue = Queue()
is_executing = False


def execute_command_queue():
    """큐에 쌓인 로봇 동작 명령을 순차적으로 실행"""
    global is_executing
    while True:
        if not command_queue.empty():
            is_executing = True
            command = command_queue.get()

            try:
                cmd = command["direction"]
                speed = command.get("speed", 1.0)
                duration = command.get("duration", 1.0)

                print(f"명령 실행: {cmd}, 속도: {speed}, 지속시간: {duration}초")

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

                # stop은 즉시 멈추는 명령이라 지속시간 없이 끝나도 됨
                if cmd != "stop":
                    steps_to_run = int(duration * 1000 / robot.getBasicTimeStep())
                    for _ in range(steps_to_run):
                        robot.step(timestep)
                    move_stop(left_wheel, right_wheel)

            except Exception as e:
                print(f"명령 실행 중 오류 발생: {e}")
                move_stop(left_wheel, right_wheel)

            finally:
                command_queue.task_done()
                is_executing = False
        else:
            time.sleep(0.1)  # 큐가 비어 있으면 잠깐 쉼


# ---------------- Function Calling 사양 ----------------

functions = [
    {
        "name": "move_robot",
        "description": "로봇을 이동시킵니다. 단일 동작이든 연속 동작이든 모두 이 함수를 사용합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "direction": {
                                "type": "string",
                                "enum": ["forward", "backward", "left", "right", "stop"],
                                "description": "이동 방향"
                            },
                            "speed": {
                                "type": "number",
                                "minimum": 0.1,
                                "maximum": 2.5,
                                "default": 1.0,
                                "description": "이동 속도 (0.1 ~ 2.5)"
                            },
                            "duration": {
                                "type": "number",
                                "minimum": 0.1,
                                "maximum": 10.0,
                                "default": 1.0,
                                "description": "이동 지속시간 (초)"
                            }
                        },
                        "required": ["direction"]
                    },
                    "description": "실행할 동작들의 배열. 단일 동작이면 배열에 하나만 넣으면 됩니다."
                }
            },
            "required": ["actions"]
        }
    }
]


def process_function_call(function_name, arguments):
    """LLM이 호출한 함수 내용(arguments)을 실제 큐에 반영"""
    try:
        if function_name == "move_robot":
            actions = arguments.get("actions", [])
            for action in actions:
                command_queue.put(action)
                print(f"동작 명령 추가: {action}")

            return (
                f"명령이 큐에 추가되었습니다. 총 {len(actions)}개 동작, "
                f"현재 큐 크기: {command_queue.qsize()}"
            )

        return "알 수 없는 함수 호출이 요청되었습니다."

    except Exception as e:
        print(f"Function call 처리 중 오류: {e}")
        return f"명령 처리 중 오류가 발생했습니다: {e}"


# ---------------- LLM 호출 래퍼 ----------------

def handle_llm_function_calling(user_message):
    """사용자 자연어 명령 → LLM → Function Calling → 큐 적재"""
    if client is None:
        print("OpenAI 클라이언트가 없습니다.")
        return "OpenAI 클라이언트가 초기화되지 않았습니다."

    try:
        messages = [
            {
                "role": "system",
                "content": (
                    "너는 로봇 제어 시스템의 AI Agent야.\n"
                    "사용자의 자연어 명령을 받아서 move_robot 함수를 호출하여 로봇을 제어해.\n\n"
                    "move_robot 함수는 actions 배열을 받아.\n"
                    "\"앞으로 가줘\" → actions: [{\"direction\": \"forward\"}]\n"
                    "\"앞으로 가다가 왼쪽으로 회전해줘\" → "
                    "actions: [{\"direction\": \"forward\"}, {\"direction\": \"left\"}]\n"
                    "\"뒤로 천천히 가줘\" → "
                    "actions: [{\"direction\": \"backward\", \"speed\": 0.5}]\n\n"
                    "로봇에 대한 이동 명령이 있으면 반드시 move_robot 함수를 호출해."
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        print(f"LLM Function Calling 요청: {user_message}")

        response = client.chat.completions.create(
            model="gpt-",  # 최신 reasoning 모델 (환경에 맞게 수정 가능)
            messages=messages,
            functions=functions,
            function_call="auto",
            max_completion_tokens=200,
            timeout=15
        )

        message = response.choices[0].message

        if hasattr(message, "function_call") and message.function_call:
            function_name = message.function_call.name
            function_args = json.loads(message.function_call.arguments)

            print(f"함수 호출: {function_name}")
            print(f"함수 인수: {function_args}")

            result = process_function_call(function_name, function_args)
            return result

        # 함수 호출이 아닌 단순 답변 (예: "이미 정지중입니다")
        return message.content if getattr(message, "content", None) else "명령을 처리할 수 없습니다."

    except Exception as e:
        print(f"LLM Function Calling 오류: {e}")
        return f"명령 처리 중 오류가 발생했습니다: {e}"


# ---------------- 초기화 (env, OpenAI, Webots) ----------------

dotenv.load_dotenv()  # .env에서 OPENAI_API_KEY 로드

try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    print("OpenAI 클라이언트 초기화 완료")
except Exception as e:
    print(f"OpenAI 클라이언트 초기화 실패: {e}")
    print("API 키가 없거나 네트워크 문제일 수 있습니다.")
    client = None

robot = Robot()
timestep = int(robot.getBasicTimeStep())
print(f"기본 시간 스텝: {timestep} ms")

# 로봇 노드 (주의: 일반 Robot 컨트롤러면 getSelf() 안 됨)
# Supervisor 컨트롤러에서 자기 자신이 아니라 특정 로봇을 추적하려면
# world의 DEF 이름으로 getFromDef("MY_ROBOT") 써야 할 수도 있음.
try:
    robot_node = robot.getSelf()
except Exception:
    robot_node = None

# 바퀴 디바이스
left_wheel = robot.getDevice("MLW")
right_wheel = robot.getDevice("MRW")

# 명령 실행 스레드 시작
command_thread = threading.Thread(target=execute_command_queue, daemon=True)
command_thread.start()
print("명령 큐 처리 스레드 시작됨")


# ---------------- 메인 루프 ----------------

step = 0
while robot.step(timestep) != -1:
    step += 1

    # 상태 출력 (10 스텝마다)
    if step % 10 == 0:
        if robot_node is not None:
            try:
                position = robot_node.getPosition()
                print(f"로봇 위치: {position}")
            except Exception:
                pass

        if command_queue.qsize() > 0:
            print(f"현재 큐 크기: {command_queue.qsize()}, 실행 중: {is_executing}")

    # Webots ↔ 브라우저 메시지 수신
    message = robot.wwiReceiveText()
    if message:
        print('USER_MESSAGE: ' + message)

        result = handle_llm_function_calling(message)
        print(f"Function Calling 결과: {result}")

        reply = (
            f"명령 처리 결과: {result}\n"
            f"현재 큐 크기: {command_queue.qsize()}"
        )
        reply = html_format(reply)
        robot.wwiSendText(reply)
