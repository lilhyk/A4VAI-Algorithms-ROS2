# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node  # rclpy의 node class 사용

from std_msgs.msg import String  # topic을 지나가는 data를 구조화하기 위해 built-in string message type import

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import cv2
import numpy as np

# 위 줄은 node의 dependencies를 나타내며, package.xml에 포함되어야 함

class MinimalPublisher(Node):  # MinimalPublisher는 Node로부터 상속됨 (Node의 subclass)

    def __init__(self):
        super().__init__('minimal_publisher')  # Node class의 custructor를 부르고 minimal_publisher라는 node 이름 줌
        
        self.publisher_ = self.create_publisher(String, 'topic', 10)  # node가 topic으로 String 타입의 message를 publish하고 que size는 10이라고 선언 
        
        timer_period = 0.5  # seconds  # 매 0.5초마다 실행하는 callback과 함께 timer 생성
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0  # callback에서 사용되는 counter
        # self.cv_image = cv2.imread('256-001.png')  # ws_ros2 폴더 (workspace)에 있어야 함!!!
        self.bridge = CvBridge()

    def timer_callback(self):  # 더해지는 counter 값과 함께 message를 생성하고, 이를 get_logger().info와 함께 console창에 publish한다.

        # 1: 이미지 경로, 2: 시작점, 3: 도착점, 4: Mode
        msg = String() 
        msg.data = '1000-003.png'

        self.publisher_.publish(msg)
        
        self.get_logger().info('Publishing an image:')
        
        self.i += 1



def main(args=None):
    rclpy.init(args=args)  # rclpy 라이브러리 초기화

    minimal_publisher = MinimalPublisher()  # Node 생성

    rclpy.spin_once(minimal_publisher)  # Node의 callback들이 불러지도록 Node를 "spin"

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
