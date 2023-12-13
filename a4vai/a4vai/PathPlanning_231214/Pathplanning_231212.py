import numpy as np
import onnx
import onnxruntime as ort
import cv2
import time
import random

model_path = "C:\\Users\\hy\\PycharmProjects\\pythonProject\\Main Code_HeightMap_khw\\90_exp_263k.onnx"
model_path2 = "C:\\Users\\hy\\PycharmProjects\\pythonProject\\Main Code_HeightMap_khw\\test26.onnx"
image_path = "C:\\Users\\hy\\PycharmProjects\\pythonProject\\Main Code_HeightMap_khw\\heightmapsGenerated\\1000-003.png"


## Range [-2500, 2500]으로 바꾸기
MapSize = 1000  # size 500
Step_Num_custom = MapSize + 1000
Init_custom = np.array([100, 2, 100])
Target_custom = np.array([950, 2, 950])

Waypoint2 = np.zeros((Step_Num_custom,3))
Waypoint = np.zeros((Step_Num_custom, 3))
Obs_mat = np.array([[]])
Waypoint[0] = Init_custom[:]

class PathPlanning:
    def __init__(self, model_path, image_path, map_size=1000):
        self.model = onnx.load(model_path)
        self.ort_session = ort.InferenceSession(model_path)
        self.map_size = map_size
        self.raw_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.raw_image_flipped = cv2.flip(self.raw_image, 0)
        self.image_new = np.where(self.raw_image_flipped < 130, 0, 1)


    def compute_path(self, Init, Target, step_num):
        # 변수 초기화
        ds = 0      # ds의 초기값 설정 (이 값은 문맥에 맞게 조정해야 할 수도 있음)
        Image_New =  self.image_new
        Init = np.array(Init)
        Target = np.array(Target)
        Pos = Init  # 초기 위치로 Pos 설정

        # 경로 계산
        waypoint = np.zeros((step_num, 3))
        for i in range(1, step_num):
            ## Initialize
            Obs1 = np.zeros((1, 3))
            Obs2 = np.zeros((1, 3))
            Obs3 = np.zeros((1, 12))

            ## Make Obs3(Lidar Terms)
            MaxLidar = 100  # Lidar의 최대 거리 비율 조정 factor
            scalefactor = 20  # map size 비율 조정 factor




            for j in range(0, 12):  # 0 - 11
                LidarState = Pos + ds
                for k in range(1, 5):
                    LidarState = LidarState + scalefactor * np.array(
                        [np.cos(((30) * (j - 1)) * np.pi / 180), 0,
                         np.sin(((30) * (j - 1)) * np.pi / 180)])  # Radian으로 해야 함

                    # Map Size 넘어갈만큼 진전시켰으면 Obs3 0으로 하고 Data 수집 멈추기
                    if LidarState[0] >= MapSize or LidarState[2] >= MapSize:
                        Diff = LidarState - (Pos + 1 * ds)
                        if np.linalg.norm(Diff) <= MaxLidar:
                            Obs3[0][j] = (np.linalg.norm(Diff)) / scalefactor  # Scaling 필요
                            break
                        else:
                            Obs3[0][j] = MaxLidar / scalefactor
                            break

                    if LidarState[0] < 0 or LidarState[2] < 0:
                        Diff = LidarState - (Pos + 1 * ds)
                        if np.linalg.norm(Diff) <= MaxLidar:
                            Obs3[0][j] = (np.linalg.norm(Diff)) / scalefactor  # s Scaling 필요
                            break
                        else:
                            Obs3[0][j] = MaxLidar / scalefactor
                            break

                    if Image_New[int(LidarState[2])][int(LidarState[0])] > 0:  # 꼭 후처리된 Map에서 # 안의 순서 주의
                        Diff = LidarState - (Pos + 1 * ds)
                        if np.linalg.norm(Diff) <= MaxLidar:
                            Obs3[0][j] = (np.linalg.norm(Diff)) / scalefactor  # Scaling 필요
                            break
                        else:
                            Obs3[0][j] = MaxLidar / scalefactor
                            break

                    Obs3[0][j] = MaxLidar / scalefactor

            # Observation for onnx range [-2500, 2500]
            # Target[0] = Target[0] - 2500
            # Target[2] = Target[2] - 2500
            # Pos[0] = Pos[0] - 2500
            # Pos[2] = Pos[2] - 2500
            Obs1 = (Target - Init) / np.linalg.norm(Target - Init)
            Obs2 = (Target - Pos) / scalefactor

            ## Make Observation
            Obs = np.random.randn(1, 18)
            Obs[0][0] = Obs1[0]
            Obs[0][1] = Obs1[1]
            Obs[0][2] = Obs1[2]
            Obs[0][3] = Obs2[0]
            Obs[0][4] = Obs2[1]
            Obs[0][5] = Obs2[2]

            Obs[0][6] = Obs3[0][0]
            Obs[0][7] = Obs3[0][1]
            Obs[0][8] = Obs3[0][2]
            Obs[0][9] = Obs3[0][3]
            Obs[0][10] = Obs3[0][4]
            Obs[0][11] = Obs3[0][5]
            Obs[0][12] = Obs3[0][6]
            Obs[0][13] = Obs3[0][7]
            Obs[0][14] = Obs3[0][8]
            Obs[0][15] = Obs3[0][9]
            Obs[0][16] = Obs3[0][10]
            Obs[0][17] = Obs3[0][11]


            Act = self.ort_session.run(None, {"obs_0": Obs.astype(np.float32)})

            ## Make Move
            Act = Act[2][0][0]  # np.linalg.norm(Act[2][0] - Act[2][1])

            for j in range(0, 5):
                # LOS 벡터 계산
                LOS_2D_N = (Target - Pos) / np.linalg.norm(Target - Pos)

                # Act 값을 사용하여 액션 각도 계산
                if Act > 0:
                    Action_angle = (90 * np.exp(5 * Act) / (np.exp(5) - 1) - 90 / (np.exp(5) - 1)) * np.pi / 180
                elif Act <= 0:
                    Action_angle = (-90 * np.exp(5 * (-Act)) / (np.exp(5) - 1) + 90 / (np.exp(5) - 1)) * np.pi / 180

                # 액션 벡터 계산
                Action_Vec = np.array([
                    LOS_2D_N[0] * np.cos(Action_angle) + LOS_2D_N[2] * np.sin(Action_angle),
                    0,
                    -LOS_2D_N[0] * np.sin(Action_angle) + LOS_2D_N[2] * np.cos(Action_angle)
                ])

                # 이동 벡터 계산 및 적용
                dS = 5 * Action_Vec

                Waypoint2[i][0] = dS[0]
                Waypoint2[i][1] = dS[1]
                Waypoint2[i][2] = dS[2]



                Pos = Pos + dS

            # Observation for onnx range [-2500, 2500]
            # Target[0] = Target[0] + 2500
            # Target[2] = Target[2] + 2500
            # Pos[0] = Pos[0] + 2500
            # Pos[2] = Pos[2] + 2500

            ## Set End Condtion
            if (np.linalg.norm(Target - Pos) < 25):
                break

            Waypoint[i][0] = Pos[0]
            Waypoint[i][1] = Pos[1]
            Waypoint[i][2] = Pos[2]


            # 경로 저장
            self.path_x = Waypoint[:i + 1, 0]
            self.path_y = Waypoint[:i + 1, 2]

    def plot_binary(self, output_path, step_num):
        # 결과 이미지 생성 및 저장
        path_x = self.path_x
        path_y = self.path_y

        Image_New = self.image_new
        Image_New2 = Image_New * 255
        Image_New2 = np.uint8(np.uint8((255 - Image_New2)))

        # Image_New2 = cv2.flip(Image_New2, 0)
        Image_New2 = cv2.flip(Image_New2, 1)
        Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        # Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        # Image_New2 = cv2.rotate(Image_New2, cv2.ROTATE_90_CLOCKWISE)
        imageLine = Image_New2.copy()
        # 이미지 크기에 따른 그리드 간격 설정
        grid_interval = 20

        # Image_New2 이미지에 그리드 그리기
        for x in range(0, imageLine.shape[1], grid_interval):  # 이미지의 너비에 따라
            cv2.line(imageLine, (x, 0), (x, imageLine.shape[0]), color=(125, 125, 125), thickness=1)

        for y in range(0, imageLine.shape[0], grid_interval):  # 이미지의 높이에 따라
            cv2.line(imageLine, (0, y), (imageLine.shape[1], y), color=(125, 125, 125), thickness=1)

        # 이미지에 맞게 SAC Waypoint 변경 후 그리기
        for i in range(1, step_num - 1):  # Changed to step_num - 1
            for m in range(0, len(self.path_x) - 2):
                Im_i = int(self.path_x[m + 1])
                Im_j = MapSize - int(self.path_y[m + 1])

                Im_iN = int(self.path_x[m + 2])
                Im_jN = MapSize - int(self.path_y[m + 2])

                # 각 웨이포인트에 점 찍기 (thickness 2)
                cv2.circle(imageLine, (Im_i, Im_j), radius=2, color=(0, 255, 0), thickness=2)

                # 웨이포인트 사이를 선으로 연결 (thickness 1)
                cv2.line(imageLine, (Im_i, Im_j), (Im_iN, Im_jN), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(output_path, imageLine)  ################################
        cv2.imshow('Binary Path Image', imageLine)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def plot_original(self, output_path, step_num):
        ## Plot and Save Image
        path_x = self.path_x
        path_y = self.path_y

        imageLine2 = self.raw_image

        # 이미지에 맞게 SAC Waypoint 변경 후 그리기
        for i in range(1, step_num - 1):  # Changed to step_num - 1
            for m in range(0, len(self.path_x) - 2):
                Im_i = int(self.path_x[m + 1])
                Im_j = MapSize - int(self.path_y[m + 1])

                Im_iN = int(self.path_x[m + 2])
                Im_jN = MapSize - int(self.path_y[m + 2])

                # 각 웨이포인트에 점 찍기 (thickness 2)
                cv2.circle(imageLine2, (Im_i, Im_j), radius=2, color=(0, 255, 0), thickness=2)

                # 웨이포인트 사이를 선으로 연결 (thickness 1)
                cv2.line(imageLine2, (Im_i, Im_j), (Im_iN, Im_jN), (0, 255, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite(output_path, imageLine2)  ################################
        cv2.imshow('Binary Path Image', imageLine2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def total_waypoint_distance(self):
        total_distance = 0
        for i in range(1, len(self.path_x)):
            dx = self.path_x[i] - self.path_x[i - 1]
            dy = self.path_y[i] - self.path_y[i - 1]
            total_distance += np.sqrt(dx**2 + dy**2)
        return total_distance

    def init_to_target_distance(self):
        dx = self.path_x[-1] - self.path_x[0]
        dy = self.path_y[-1] - self.path_y[0]
        return np.sqrt(dx**2 + dy**2)

    def print_distance_ratio(self):
        total_wp_distance = self.total_waypoint_distance()
        init_target_distance = self.init_to_target_distance()
        # 절대오차 계산
        absolute_error = abs(total_wp_distance - init_target_distance)

        # 상대오차 계산 (0으로 나누는 경우 예외 처리)
        if init_target_distance != 0:
            relative_error = absolute_error / init_target_distance
            print(f"절대오차: {absolute_error:.2f}, 상대오차: {relative_error:.2%}")
        else:
            print(f"절대오차: {absolute_error:.2f}, 상대오차: 계산 불가 (분모가 0)")

        #ratio = ((total_wp_distance-init_target_distance / total_wp_distance) )*100
        #print("Total Waypoint Distance / Init to Target Distance: {:.2f}%".format(ratio))


# model 90 deg
planner = PathPlanning(model_path, image_path)
planner.compute_path(Init_custom, Target_custom, Step_Num_custom) # start point , target point,step_num, max lidar, scale factor
planner.plot_binary("MCResult/path_test_001_binary.png", Step_Num_custom)
planner.plot_original("MCResult/path_test_001_og.png", Step_Num_custom)
planner.print_distance_ratio()

# model test26
planner2 = PathPlanning(model_path2,image_path)
planner.compute_path(Init_custom, Target_custom, Step_Num_custom) # start point , target point,
planner.plot_binary("MCResult/path_test_002_binary.png", Step_Num_custom)
planner.plot_original("MCResult/path_test_002_og.png", Step_Num_custom)
planner.print_distance_ratio()






