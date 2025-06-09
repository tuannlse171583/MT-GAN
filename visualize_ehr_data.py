# visualize_ehr_data.py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess.auxiliary import real_data_stat, generate_code_code_adjacent
from preprocess.parse_csv import Mimic3Parser
from preprocess.encode import encode_concept
from preprocess.build_dataset import build_real_data

# === Bước 1: Parse dữ liệu từ MIMIC-III ===
data_path = '/content/MT-GAN/data/mimic3/raw'
parser = Mimic3Parser(data_path)
patient_admission, admission_codes = parser.parse(sample_num=1000, seed=42)  # sample 1000 bệnh nhân

# === Bước 2: Mã hóa mã bệnh ===
admission_codes_encoded, code_map = encode_concept(patient_admission, admission_codes)

# === Bước 3: Xây dựng dữ liệu thực ===
code_num = len(code_map)
max_admission_num = max(len(adms) for adms in patient_admission.values())
pids = list(patient_admission.keys())
real_data_x, lens = build_real_data(pids, patient_admission, admission_codes_encoded, max_admission_num, code_num)

# === Bước 4: Tính thống kê ===
admission_dist, code_visit_dist, code_patient_dist = real_data_stat(real_data_x, lens)

# === Biểu đồ 1: Phân bố số lần nhập viện (chi tiết từng mốc 1, 2, 3...) ===
plt.figure(figsize=(10, 5))
x_ticks = np.arange(1, len(admission_dist) + 1)
plt.bar(x_ticks, admission_dist, color='skyblue', edgecolor='black')

plt.xticks(x_ticks)
plt.xlabel("Số lần nhập viện")
plt.ylabel("Tỷ lệ bệnh nhân")
plt.title("Phân bố số lần nhập viện của bệnh nhân")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("admission_distribution_detailed.png")
plt.show()

# === Biểu đồ 2: Phân bố số lượng bệnh nhân theo số lần nhập viện (hiển thị số cụ thể) ===
plt.figure(figsize=(10, 5))
patient_counts = admission_dist * len(lens)

bars = plt.bar(x_ticks, patient_counts, color='skyblue', edgecolor='black')
plt.xticks(x_ticks)
plt.xlabel("Số lần nhập viện")
plt.ylabel("Số bệnh nhân")
plt.title("Số lượng bệnh nhân theo số lần nhập viện")
plt.grid(axis='y', linestyle='--', alpha=0.7)

for bar, count in zip(bars, patient_counts):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, height + 1, f'{int(count)}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig("admission_distribution_counts.png")
plt.show()

# === Biểu đồ 3: Biểu đồ hình tròn đã gom nhóm nhỏ (<1%) thành "Khác" ===
plt.figure(figsize=(7, 7))
threshold = 0.01  # 1%
large_idx = admission_dist >= threshold
small_idx = ~large_idx

labels_main = [f"{i+1} lần" for i in range(len(admission_dist)) if large_idx[i]]
sizes_main = admission_dist[large_idx]

if small_idx.any():
    labels_main.append("Khác")
    sizes_main = np.append(sizes_main, admission_dist[small_idx].sum())

plt.pie(sizes_main, labels=labels_main, autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title("Tỷ lệ phần trăm bệnh nhân theo số lần nhập viện (gom nhóm nhỏ)")
plt.tight_layout()
plt.savefig("admission_distribution_pie_grouped.png")
plt.show()
