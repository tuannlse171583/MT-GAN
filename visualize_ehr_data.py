# visualize_ehr_data.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess.auxiliary import real_data_stat, generate_code_code_adjacent

# === Thay đổi phần dưới theo dữ liệu thực tế ===
# Giả sử bạn đã có sẵn các biến sau từ quá trình xử lý dữ liệu:
# real_data_x: ndarray (n_benh_nhan, max_luot_kham, so_ma_benh)
# lens: ndarray (n_benh_nhan,) - số lượt khám của từng bệnh nhân
# pids: danh sách patient ids
# patient_admission: dict pid -> list[admission]
# admission_codes_encoded: dict adm_id -> list[int] (mã bệnh đã mã hóa)
# code_num: tổng số mã bệnh

# ==== Ví dụ mẫu: Thay bằng dữ liệu thật ====
# from your_data_loader import real_data_x, lens, pids, patient_admission, admission_codes_encoded, code_num

# === Tính thống kê ===
admission_dist, code_visit_dist, code_patient_dist = real_data_stat(real_data_x, lens)

# === Biểu đồ 1: Phân bố số lần nhập viện ===
plt.figure(figsize=(6, 4))
plt.bar(range(1, len(admission_dist) + 1), admission_dist)
plt.xlabel("Số lần nhập viện")
plt.ylabel("Tỷ lệ bệnh nhân")
plt.title("Phân bố số lần nhập viện")
plt.tight_layout()
plt.savefig("admission_distribution.png")
plt.show()
