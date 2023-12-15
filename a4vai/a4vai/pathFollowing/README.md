# Path Following Algorithms

## 1. 개요 및 구조

- 기본 유도 법칙 + MPPI 유도 알고리즘의 구현
- `test_att_ctrl.py` 가 main문에 해당하는 ROS2 node 부분
- 나머지 파일들(`.py`)은 필요한 클래스, 함수 등을 포함한 파일들

# 2. 필요 작업

- **MPPI 관련 create_timer 부분의 분리가 필요**

- [`test_att_ctrl.py`](test_att_ctrl.py#L144)

```python
# callback test_attitude_control
period_MPPI_param = 0.05
self.timer = self.create_timer(period_MPPI_param, self.PF_MPPI_param, callback_group = self.MPPIGroup)
```

- **지금과 같이 하나의 노드에서 돌리면, ROS2가 MPPI 명령을 계산할 때 문제가 발생**
    - ROS2 to PX4의 자세제어 명령이 전달되지 못하고 끊어지는 듯한 현상으로, 제대로 날지 못 함.
- **통합 시뮬레이션 환경 구성에서 이 부분을 해결해야 함.**
    - 쓰레드분리나, 노드 분리 등의 작업이 필요할 것으로 보임.
- 지금 상태에서는 하나의 노드에 모든 기능이 다 구현되어있으므로, 제대로 날지 못함.
- 위의 MPPI 관련 create_timer 부분을 주석처리하면 잘 날 것으로 보임.

<br/>

- **작업 중인 사항들:**
    - 나머지 파일들(.py) 부분 내용 정리 필요
    - GP 관련 모듈 적용 필요