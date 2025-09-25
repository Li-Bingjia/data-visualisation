import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import numpy as np



df = pd.read_csv("ncr_ride_bookings.csv")
print(df.head())

# 2. 只保留数值型列
numeric_df = df.select_dtypes(include="number")

# 删除包含 Cancelled 或 Incomplete 的列
df = df.drop(columns=[col for col in df.columns if "cancelled" in col.lower() or "incomplete" in col.lower()],
             errors="ignore")

# 计算相关性矩阵
corr = df.corr(numeric_only=True)

# 创建子图：1 行 3 列
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# -------------------------------
# 图 1: 相关性热力图
# -------------------------------
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, ax=axes[0])
axes[0].set_title("Correlation Heatmap of Ride Metrics", fontsize=12, fontweight="bold")

# -------------------------------
# 图 2: 散点图
# -------------------------------
axes[1].scatter(df["Avg VTAT"], df["Customer Rating"], alpha=0.3, s=10, c="blue")
axes[1].set_xlabel("Average VTAT (Waiting Time)")
axes[1].set_ylabel("Customer Rating")
axes[1].set_title("Customer Rating vs Waiting Time", fontsize=12, fontweight="bold")
axes[1].grid(alpha=0.3)

# -------------------------------
# 图 3: 小提琴图
# -------------------------------
sns.violinplot(x="Customer Rating", y="Avg CTAT", data=df, inner="quartile", ax=axes[2])
axes[2].set_title("Distribution of CTAT Across Customer Ratings", fontsize=12, fontweight="bold")



# 调整整体布局
plt.tight_layout()
plt.show()

# 读取 CSV
df = pd.read_csv("ncr_ride_bookings.csv")
data = df["Avg VTAT"].dropna().tolist()

# 数据归一化
data_norm = (np.array(data) - np.min(data)) / (np.max(data) - np.min(data))

# ==== 平滑插值 ====
frames_per_step = 10
interp_data = []
for i in range(len(data_norm) - 1):
    start, end = data_norm[i], data_norm[i + 1]
    interp = np.linspace(start, end, frames_per_step, endpoint=False)
    interp_data.extend(interp)
interp_data = np.array(interp_data)

# 画布
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.axis("off")

x = np.linspace(-3, 3, 400)

def rotate(x, y, angle):
    """二维旋转变换"""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a
    return x_new, y_new

def update(frame):
    ax.clear()
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.axis("off")

    norm_val = interp_data[frame % len(interp_data)]

    # 基础曲线
    y = np.sin(x * (1 + norm_val * 2) + frame * 0.05) * (0.5 + norm_val)

    # 曲线旋转（保持原速度）
    angle_line = frame * 0.2
    x_rot, y_rot = rotate(x, y, angle_line)

    # 曲线绘制
    for i in range(1, 20):
        alpha = 0.5 / i
        offset = i * 0.5 * norm_val
        ax.fill_between(x_rot, y_rot + offset, y_rot - offset,
                        color=plt.cm.cividis(norm_val),
                        alpha=alpha)
    
    # 圆环扩散 + 更快的旋转
    theta = np.linspace(0, 2*np.pi, 300)
    radius = 0.02 + norm_val * 1.5 + (frame % 40) * 0.08
    cx, cy = radius * np.cos(theta), radius * np.sin(theta)

    angle_circle = frame * 0.08  # ⚡ 圆的旋转速度更快
    cx, cy = rotate(cx, cy, angle_circle)

    ax.fill(cx, cy, color=plt.cm.magma(norm_val), alpha=0.1)

# 动画
ani = FuncAnimation(fig, update, frames=len(interp_data), interval=100)
plt.show()
