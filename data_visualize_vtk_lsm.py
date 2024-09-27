import numpy as np
import vtk
import os

# 파일 경로 설정
base_dir = 'C:/Users/LSM/Desktop/class_mobility_A/autopilot/3d_mod_av_db'  # 데이터의 기본 디렉토리 경로
labels_dir = os.path.join(base_dir, 'labels')  # 라벨 파일이 저장된 디렉토리
points_dir = os.path.join(base_dir, 'points')  # 포인트 클라우드 파일이 저장된 디렉토리
output_dir = os.path.join(base_dir, 'visualization_vtk_lsm')  # 시각화된 이미지를 저장할 디렉토리

# 출력 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)  # 출력 디렉토리가 존재하지 않으면 생성

# 바운딩 박스의 코너를 계산하는 함수
def get_bbox_corners(x, y, z, l, w, h, yaw):
    """
    바운딩 박스의 8개의 코너 좌표를 계산합니다.
    :param x, y, z: 바운딩 박스 중심의 x, y, z 좌표
    :param l, w, h: 바운딩 박스의 길이(length), 너비(width), 높이(height)
    :param yaw: 바운딩 박스의 회전 각도 (yaw)
    :return: 변환된 바운딩 박스 코너 좌표
    """
    corners = np.array([
        [l / 2, w / 2, h / 2],
        [l / 2, -w / 2, h / 2],
        [-l / 2, -w / 2, h / 2],
        [-l / 2, w / 2, h / 2],
        [l / 2, w / 2, -h / 2],
        [l / 2, -w / 2, -h / 2],
        [-l / 2, -w / 2, -h / 2],
        [-l / 2, w / 2, -h / 2],
    ])

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    rotation_matrix = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw, cos_yaw, 0],
        [0, 0, 1]
    ])

    rotated_corners = corners @ rotation_matrix.T
    translated_corners = rotated_corners + np.array([x, y, z])

    return translated_corners

# 범례 색상 설정
legend_colors = {
    'vehicle': (0, 0, 1),      # 파란색 (차)
    'pedestrian': (1, 0, 0),   # 빨간색 (보행자)
    'cyclist': (1, 1, 0),      # 노란색 (사이클)
    'unknown': (0.5, 0.5, 0.5) # 회색 (모르는 객체)
}

def visualize_with_vtk(points, labels, output_image_path):
    """
    VTK를 사용하여 포인트 클라우드와 바운딩 박스를 시각화합니다.
    데이터 수집 차량을 화면 정중앙에 배치합니다.
    """
    # 데이터 수집 차량의 위치를 찾습니다 (일반적으로 원점에 가장 가까운 차량)
    ego_vehicle = min(labels, key=lambda l: l['x']**2 + l['y']**2 + l['z']**2)
    ego_center = np.array([ego_vehicle['x'], ego_vehicle['y'], ego_vehicle['z']])

    # 모든 점과 라벨을 ego 차량 중심으로 이동
    points[:, :3] -= ego_center  # 포인트 클라우드의 첫 세 열만 이동
    for label in labels:
        label['x'] -= ego_center[0]
        label['y'] -= ego_center[1]
        label['z'] -= ego_center[2]

    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.SetOffScreenRendering(1)
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)  # 해상도 유지

    # 포인트 클라우드 추가
    point_cloud = vtk.vtkPoints()
    for point in points:
        point_cloud.InsertNextPoint(point[:3])

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(point_cloud)

    vertex_glyph_filter = vtk.vtkVertexGlyphFilter()
    vertex_glyph_filter.SetInputData(poly_data)
    vertex_glyph_filter.Update()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(vertex_glyph_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(0.5, 0.5, 0.5)
    actor.GetProperty().SetPointSize(1)

    renderer.AddActor(actor)

    # 바운딩 박스 추가
    for label in labels:
        corners = get_bbox_corners(label['x'], label['y'], label['z'], label['l'], label['w'], label['h'], label['yaw'])
        bbox_points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        for corner in corners:
            bbox_points.InsertNextPoint(corner)

        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        for edge in edges:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, edge[0])
            line.GetPointIds().SetId(1, edge[1])
            lines.InsertNextCell(line)

        bbox_poly_data = vtk.vtkPolyData()
        bbox_poly_data.SetPoints(bbox_points)
        bbox_poly_data.SetLines(lines)

        bbox_mapper = vtk.vtkPolyDataMapper()
        bbox_mapper.SetInputData(bbox_poly_data)

        bbox_actor = vtk.vtkActor()
        bbox_actor.SetMapper(bbox_mapper)
        bbox_actor.GetProperty().SetColor(*legend_colors.get(label['class'], (0.5, 0.5, 0.5)))

        renderer.AddActor(bbox_actor)

    # 카메라 설정
    center_of_mass = np.mean(points[:, :3], axis=0)  # 포인트 클라우드의 중심 계산
    camera = renderer.GetActiveCamera()  # 활성 카메라 가져오기
    camera.SetFocalPoint(center_of_mass)  # 카메라 초점 설정
    camera.SetPosition(center_of_mass + np.array([0, -200, 200]))  # 카메라 위치 설정
    camera.SetViewUp(0, 0, 1)  # 카메라 상단 방향 설정
    camera.SetClippingRange(0.1, 1000)  # 클리핑 범위 설정

    renderer.SetBackground(1, 1, 1)  # 배경색 설정 (검정)
    renderer.ResetCameraClippingRange()  # 카메라 클리핑 범위 재설정

    # 범례 추가 (오른쪽 하단에 배치)
    def add_legend_text(text, color, position):
        legend_text = vtk.vtkTextActor()
        legend_text.SetTextScaleModeToNone()
        legend_text.SetPosition(*position)
        legend_text.GetTextProperty().SetFontSize(10)  # 글자 크기 유지
        legend_text.GetTextProperty().SetColor(*color)
        legend_text.SetInput(text)
        renderer.AddActor2D(legend_text)

    add_legend_text("Vehicle - Blue", (0, 0, 1), (1400, 60))
    add_legend_text("Pedestrian - Red", (1, 0, 0), (1400, 100))
    add_legend_text("Cyclist - Yellow", (1, 1, 0), (1400, 140))
    add_legend_text("Unknown - Gray", (0.5, 0.5, 0.5), (1400, 180))

    # 렌더링 및 이미지 저장
    render_window.Render()
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.SetScale(2)  # 이미지 해상도 스케일 조정
    window_to_image_filter.SetInputBufferTypeToRGB()
    window_to_image_filter.Update()

    writer = vtk.vtkPNGWriter()
    writer.SetFileName(output_image_path)
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    writer.Write()

    print(f"이미지 저장 완료: {output_image_path}")

# 시작할 파일 이름 설정
start_file = '00000000.npy' # 00000000부터 시작
start_processing = False

# points 및 labels 폴더 내의 모든 파일을 오름차순으로 정렬하여 순회
for point_file in sorted(os.listdir(points_dir)):
    if point_file.endswith('.npy'):
        if not start_processing:
            if point_file == start_file:
                start_processing = True
            else:
                continue

        # 포인트 클라우드 파일 경로 설정
        point_cloud_path = os.path.join(points_dir, point_file)
        label_file = point_file.replace('.npy', '.txt')
        label_path = os.path.join(labels_dir, label_file)

        if not os.path.exists(label_path):
            print(f"라벨 파일을 찾을 수 없습니다: {label_path}")
            continue

        # 포인트 클라우드 데이터 로드
        point_cloud = np.load(point_cloud_path)

        # 라벨 데이터 로드
        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                tokens = line.strip().split()
                if len(tokens) != 8:
                    print(f"형식이 잘못된 라인 건너뜀: {line}")
                    continue
                x, y, z, l, w, h, yaw, class_name = tokens
                labels.append({
                    'x': float(x),
                    'y': float(y),
                    'z': float(z),
                    'l': float(l),
                    'w': float(w),
                    'h': float(h),
                    'yaw': float(yaw),
                    'class': class_name.lower()
                })

        # 이미지 저장 경로 설정
        output_image_path = os.path.join(output_dir, point_file.replace('.npy', '.png'))
        visualize_with_vtk(point_cloud, labels, output_image_path)