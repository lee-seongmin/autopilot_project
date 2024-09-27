import cv2
import os

# 이미지 파일들이 있는 폴더 경로 설정
image_dir = 'C:/Users/LSM/Desktop/class_mobility_A/autopilot/3d_mod_av_db/visualization_vtk_lsm'  # 이미지 파일들이 있는 디렉토리 경로
output_video_dir = 'C:/Users/LSM/Desktop/class_mobility_A/autopilot'  # 비디오 파일을 저장할 디렉토리 경로
output_video_name = 'visualization_vtk_lsm_fps_down.mp4'  # 생성할 비디오 파일의 이름
output_video_path = os.path.join(output_video_dir, output_video_name)  # 전체 비디오 파일 경로 설정

# 파일 이름의 범위를 설정
start_file = 0  # 시작할 파일 번호
end_file = 210906  # 끝낼 파일 번호

# 출력 디렉토리가 없으면 생성
os.makedirs(output_video_dir, exist_ok=True)  # 비디오 파일을 저장할 디렉토리가 없을 경우 생성

# 이미지 파일을 오름차순으로 정렬
image_files = sorted([
    f for f in os.listdir(image_dir) 
    if f.endswith('.png') and start_file <= int(f.split('.')[0]) <= end_file  # 파일명이 숫자 범위 내에 있는 .png 파일만 선택
])

# 첫 번째 이미지로 프레임 크기 설정
first_image_path = os.path.join(image_dir, image_files[0])  # 첫 번째 이미지 파일 경로
frame = cv2.imread(first_image_path)  # 첫 번째 이미지를 읽어서 크기 정보를 가져옴
height, width, layers = frame.shape  # 이미지의 높이, 너비, 레이어(채널) 수 추출

# 비디오 작성기 초기화 (프레임당 20FPS 설정, 코덱: mp4v)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 비디오 코덱 설정 (mp4v 코덱 사용)
video = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))  # 비디오 작성기 초기화, 프레임 크기 설정

# 각 이미지를 읽어와서 비디오에 추가
for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)  # 각 이미지의 전체 경로 설정
    frame = cv2.imread(image_path)  # 이미지를 읽어옴
    video.write(frame)  # 읽은 이미지를 비디오에 프레임으로 추가

# 비디오 작성기 해제
video.release()  # 비디오 작성기 해제 및 파일 저장 완료
cv2.destroyAllWindows()  # 모든 OpenCV 창 닫기 (여기서는 필요하지 않지만 안전을 위해 호출)

# 완료 메시지 출력
print(f"비디오 생성 완료: {output_video_path}")
