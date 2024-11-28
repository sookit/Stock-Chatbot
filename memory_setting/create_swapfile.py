import os

def create_swapfile(size_in_gb=4):
    swapfile_path = "C:\\swapfile.sys"  # 스왑 파일 경로
    size_in_mb = size_in_gb * 1024  # GB → MB 변환

    # 스왑 파일 생성
    print(f"Creating swapfile of size {size_in_gb}GB...")
    os.system(f"fsutil file createnew {swapfile_path} {size_in_mb * 1024 * 1024}")

    # 스왑 파일 활성화
    print("Activating swapfile...")
    os.system(f"wmic pagefileset where name='{swapfile_path}' set InitialSize={size_in_mb},MaximumSize={size_in_mb}")
    print("Swapfile activated.")

create_swapfile(size_in_gb=4)
