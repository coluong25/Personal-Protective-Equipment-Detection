import os, shutil
from sklearn.model_selection import train_test_split

# Lấy danh sách ảnh
images = sorted(os.listdir("../data/annotated/images"))

# Chia train/val/test — giữ tên file để match với label
train, temp = train_test_split(images, test_size=0.2, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)  # 0.5 của 0.2 = 0.1 mỗi cái

# Tập list tên đi xuống vào vòng lặp dưới => vào lệnh copy
# Tạo folder và copy ảnh + label tương ứng
# for split, files in [("train", train), ("val", val), ("test", test)]:
#     os.makedirs(f"../split/images/{split}", exist_ok=True)
#     # exist_ok=True Không bị trả về FileExistsError: [Errno 17] File exists
#     os.makedirs(f"../split/labels/{split}", exist_ok=True)
    
#     for fname in files:
#         # Copy ảnh
#         shutil.copy(f"../data/annotated/images/{fname}", f"../split/images/{split}/{fname}")
        
#         # Copy label cùng tên, đổi đuôi sang .txt (chuẩn YOLO)
#         label = fname.rsplit(".", 1)[0] + ".txt"
#         if os.path.exists(f"../data/annotated/labels/{label}"):
#             shutil.copy(f"../data/annotated/labels/{label}", f"../split/labels/{split}/{label}")

# Vòng lặp copy và chia thư mục theo từng tập train val test
print(images)
# # Bỏ chia các class lệch ở mỗi split