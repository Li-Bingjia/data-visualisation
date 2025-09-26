import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import numpy as np
import random


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
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)
ax.axis("off")

# 粒子容器
particles = []

class Particle:
    def __init__(self, x, y, norm_val):
        self.x = x + random.uniform(-0.2, 0.2)  # 粒子初始位置带偏移
        self.y = y + random.uniform(-0.2, 0.2)
        self.size = random.uniform(10, 40)      # 初始大小
        self.alpha = 0.9                        # 初始透明度
        self.color = plt.cm.plasma(norm_val)    # 颜色和数据挂钩
        self.vx = random.uniform(-0.05, 0.05)   # x方向漂浮速度
        self.vy = random.uniform(-0.05, 0.05)   # y方向漂浮速度
    
    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.alpha -= 0.02   # 慢慢消失
        self.size *= 0.97    # 逐渐缩小
        return self.alpha > 0

def rotate(x, y, angle):
    """二维旋转变换"""
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x_new = x * cos_a - y * sin_a
    y_new = x * sin_a + y * cos_a
    return x_new, y_new

def update(frame):
    ax.clear()
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.axis("off")

    norm_val = interp_data[frame % len(interp_data)]

    # ========= 曲线部分（渐变烟雾线条） =========
    x = np.linspace(-3, 3, 400)
    y = np.sin(x * (1 + norm_val * 2) + frame * 0.05) * (0.5 + norm_val)

    angle_line = frame * 0.2
    x_rot, y_rot = rotate(x, y, angle_line)

    for i in range(1, 20):
        alpha = 0.2 / i
        offset = i * 0.2 * norm_val
        ax.fill_between(x_rot, y_rot + offset, y_rot - offset,
                        color=plt.cm.plasma(norm_val),
                        alpha=alpha)

    # ========= 圆心位置（呼吸运动） =========
    breathing = 0.8 * (0.5 + 0.5 * np.sin(frame * 0.1))
    radius = 0.5 + norm_val * 1.2 + breathing
    circle_x = 1.5 * np.sin(frame * 0.02)
    circle_y = 1.0 * np.cos(frame * 0.015)

    # ========= 粒子效果（圆心释放） =========
    for _ in range(4):  # 每帧释放的粒子数量
        particles.append(Particle(circle_x, circle_y, norm_val))

    alive_particles = []
    xs, ys, sizes, colors, alphas = [], [], [], [], []
    for p in particles:
        if p.update():
            alive_particles.append(p)
            xs.append(p.x)
            ys.append(p.y)
            sizes.append(p.size)
            colors.append(p.color)
            alphas.append(p.alpha)
    particles[:] = alive_particles

    ax.scatter(xs, ys, s=sizes, c=colors, alpha=alphas, edgecolors="none")

    # ========= 当前圆（半透明能量核） =========
    theta = np.linspace(0, 2*np.pi, 200)
    cx, cy = radius * np.cos(theta) + circle_x, radius * np.sin(theta) + circle_y
    ax.fill(cx, cy, color=plt.cm.magma(norm_val), alpha=0.05)

# 动画
ani = FuncAnimation(fig, update, frames=len(interp_data), interval=100)
plt.show()
