Path Planning Algorithms
========================

1\. 개요 및 구조
-----------

이 프로젝트는 기존의 경로 계획 알고리즘을 통합에 맞도록 업데이트한 버전. `Image2Plan.py` 파일은 이미지 경로를 입력으로 받아, `Plan2WP.py`에서 경로 계획을 진행.

*   **Image2Plan.py**: 이미지 경로를 메시지로 전송.
*   **Plan2WP.py**: 경로 계획 결과를 PNG 그림 파일로 생성, 이전 알고리즘과 비교하여 계산된 Cost를 출력.

2\. 필요 작업
---------

프로젝트를 정상적으로 실행하기 위해서는 ROS2 workspace 내에 'pathPlanning\_data' 폴더가 필요. 이 폴더는 다음과 같은 내용을 포함해야 합니다:

*   `src` 폴더
*   `90_exp_263k.onnx`
*   `1000-001.png`
*   `test26.onnx`
*   **pathPlanning_data** 폴더는 NAS 유창경교수님 폴더에 있음.



프로젝트를 실행하는 커맨드는 다음과 같음.

**터미널 - 1: 빌드**

`colcon build --packages-select py_pubsub`

**터미널 - 2: 이미지 경로 Publish**

`source install/setup.bash` ,

`ros2 run py_pubsub talker`

**터미널 - 3: 이미지 경로 Subscribe & 경로계획 Main문**

`source install/setup.bash`,

`ros2 run py_pubsub listener`

3\. 향후 계획
---------

향후 `Image2Plan.py`는 시작점, 도착점, Mode 등을 메시지로 변경할 예정. ,
이를 통해 더욱 다양한 시나리오에 대응할 수 있는 경로 계획 시스템을 구현할 계획.
