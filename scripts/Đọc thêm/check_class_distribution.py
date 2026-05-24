from collections import Counter
# Tần suất xuất hiện phần tử trong list


        for label_file in label_dir.glob("*.txt"):
            # glob liệt kê file/folder thoả mãn tên =>output: list
            # glob +label_dir phía trước => danh sách path, chú ý label_dir gọi glob như hàm (pathlib.Path.glob())
            with open(label_file) as f:
                # Cấu trúc with as đi với class (open) có dunder __enter__ mở file và __exit__ đóng file
                # f là object của open
                for line in f:
                    # 1 dòng trong output của f
                    line = line.strip()
                    # Bỏ \n, space, \t(tab)