### LLM-based Robot Controller 계획서 (Webots + Python + Ollama)

#### 1) 문제 정의와 목표
- **문제정의**: 자연어로 주어진 미션을 로봇 시뮬레이터(Webots) 상에서 안전하게, 순차적 행동으로 수행하는 에이전트 구현.
- **목표**: LLM이 Planner로서 고수준 계획을 수립하고, 함수콜(툴 호출)로 저수준 제어를 실행(Actor), 실패 시 재계획(Verifier)해 성공률·안전성을 확보.

#### 2) 기술 선택과 트레이드오프 (LLM 활용)
- **기본**: 로컬 LLM(Ollama, 예: Qwen2.5-7B/8B) → 지연·비용 최소화, 오프라인 가능.
- **백업**: API 모델(gpt-4o-mini 등) → 복잡한 추론/긴 컨텍스트 필요 시만 사용.
- **튜닝 포인트**: 프롬프트 체계화(System + Tool Schema), 함수콜 JSON 포맷 엄격화, `temperature` 낮게(0.2~0.4), 계획 단계는 `max_tokens` 여유.

#### 3) 시스템 아키텍처 개요
```
사용자 자연어 ─┐
               v
            [Planner (LLM)] ── 계획(steps[]) ──┐
                                               v
                               [Actor (툴 실행기, Python/Webots)] ── 센서/모터 API
                                               ^
                              상태/로그 ───────┘
               ^
검증/재계획 <─[Verifier (LLM or Rule)]  (충돌/실패 감지 시 재계획)
```
- 통신: 컨트롤러 내부에서 LLM HTTP 호출(로컬 Ollama), 또는 별도 오케스트레이터 프로세스와 소켓/HTTP.
- UI: Robot Window(`text_input`)는 디버그 메시지·상태 표시 및 수동 명령 주입에 활용.

#### 4) 에이전트 오케스트레이션
- **Planner**: 자연어 미션 → 정규화된 액션 시퀀스(JSON). 예: `go to x,y`, `avoid obstacle`, `search area` → 저수준 프리미티브로 분해.
- **Actor(툴)**: Webots API 래핑
  - `move_forward(speed, duration)`
  - `move_backward(speed, duration)`
  - `turn_left(speed, duration)` / `turn_right(speed, duration)`
  - `stop()`
  - `read_proximity()` / `read_pose()` / (선택) `read_camera()`
- **Verifier**: 규칙+LLM 혼합
  - 규칙: 충돌 임박(근접센서 임계치, 속도=0 지속, 목표 미접근) → 실패 판정
  - LLM: 로그 요약/설명, 대안 계획 제안

#### 5) 함수콜/액션 포맷(예시)
```json
{
  "mission": "주어진 공을 피해 목표 지점 (x=-0.5, y=0.3)으로 이동",
  "steps": [
    {"tool": "read_proximity"},
    {"tool": "turn_left", "speed": 2.0, "duration": 0.6},
    {"tool": "move_forward", "speed": 3.0, "duration": 2.0},
    {"tool": "read_pose"},
    {"tool": "move_forward", "speed": 3.0, "duration": 1.2},
    {"tool": "stop"}
  ]
}
```
- 속도는 `getMaxVelocity()` 기반으로 클램프, duration은 타임스텝 루프로 실행.

#### 6) 시뮬레이터(Webots) 구성
- **로봇**: E-puck (좌/우 휠 모터, 근접센서 다중). `Supervisor` 가능 시 자체 위치·자세 읽기.
- **환경**: `RectangleArena` + 장애물(Solid/공). 외부 텍스처·모델은 리포 내 자산 경로 사용.
- **센서 파이프라인**: 근접센서 활성화 및 정규화, (선택) 카메라 프레임 다운샘플링.

#### 7) 행동 실행 및 상태관리
- 메인 루프에서 현재 스텝의 종료시각을 관리(비블로킹). 스텝 완료/중단·재계획 신호 처리.
- 상태: 최근 센서값, 현재 속도/방향, 마지막 성공/실패 원인, 누적 경로.
- 로깅: `logs/rollout_YYYYMMDD.jsonl` (step, obs, action, reward-like signals).

#### 8) 실패 복구/재계획 정책
- 조건: (a) 근접센서 임계 초과, (b) 위치 변화량 미미, (c) 타임아웃.
- 조치: 정지→회피회전→재계획(Planner 호출). 재시도 횟수 상한.

#### 9) 안전 제약
- 속도: `min(left.getMaxVelocity(), right.getMaxVelocity())` 로 상한.
- 회피: 근접센서 히스테리시스 사용(노이즈 안정화).
- 경계: 아레나 경계 박스 기준 거리 임계치 유지.

#### 10) 벤치마크/지표(간단)
- **성공률**: 미션 완료 여부(목표 반경 r 이내 도달)
- **시간**: 계획+실행 총 소요(s)
- **개입**: 수동介入 횟수
- **안전위반**: 충돌 감지/경계 위반 횟수
- **지연**: Planner 호출 평균/최대 지연(ms)

#### 11) 제출물 계획
- `이름_학번.md`: 평가 항목별 적용 기술 명세서(1페이지)
- `이름_학번.mp4`: 5분 소개 영상(데모·아키텍처·지표)
- 프로젝트 폴더: 코드·월드·문서 일체(zip)

#### 12) 구현 로드맵(제출 D-일정)
- D-6~5: 오케스트레이터 스켈레톤 + 함수콜 스펙 + Webots 센서/모터 래핑
- D-4: Planner 프롬프트/툴 사용 예제 고도화, 기본 재계획
- D-3: 안전제약·로깅·리플레이, 간단 벤치마크 스크립트
- D-2: 통합 데모 시나리오·영상 스토리보드, 안정화
- D-1: 촬영·편집·문서 최종화

#### 13) 파일/모듈 구성(제안)
- `controllers/agent_controller/agent.py`: Planner/Actor/Verifier 오케스트레이션
- `controllers/agent_controller/tools.py`: Webots 액션/센서 툴 래퍼
- `controllers/agent_controller/policy.py`: 프롬프트/함수콜 제약/출력 검증
- `plugins/robot_windows/agent_ui/`: 상태 시각화·수동 명령
- `docs/`: 프롬프트, 지표, 실험노트


