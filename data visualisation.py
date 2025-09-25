import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("ncr_ride_bookings.csv")
print(df.head())

# 2. 只保留数值型列
numeric_df = df.select_dtypes(include="number")

# 3. 生成统计摘要
desc = numeric_df.describe()

# 创建 2x2 子图布局
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 删除包含 Cancelled 或 Incomplete 的列
df = df.drop(columns=[col for col in df.columns if "cancelled" in col.lower() or "incomplete" in col.lower()],
             errors="ignore")

# -------------------------------
# 图表 1: 各字段的均值柱状图
# -------------------------------
desc.loc["mean"].plot(kind="bar", ax=axes[0, 0], alpha=0.7, color="skyblue")
axes[0, 0].set_title("Average Values of Numerical Features", fontsize=14, fontweight="bold")
axes[0, 0].set_ylabel("Mean Value")
axes[0, 0].tick_params(axis="x", rotation=45)
axes[0, 0].grid(axis="y", alpha=0.3)

# -------------------------------
# 图表 2: 箱线图
# -------------------------------
numeric_df.plot(kind="box", ax=axes[0, 1])
axes[0, 1].set_title("Boxplot of Key Metrics", fontsize=14, fontweight="bold")
axes[0, 1].set_ylabel("Value")

# -------------------------------
# 图表 3: 直方图
# -------------------------------
numeric_df.hist(bins=20, ax=axes[1, 0], grid=False)
axes[1, 0].set_title("Distribution of Features", fontsize=14, fontweight="bold")
axes[1, 0].set_ylabel("Frequency")

# -------------------------------
# 图表 4: 均值 vs 中位数对比
# -------------------------------
stats = desc.loc[["mean", "50%"]]
stats.T.plot(kind="bar", ax=axes[1, 1], alpha=0.8)
axes[1, 1].set_title("Comparison of Mean and Median", fontsize=14, fontweight="bold")
axes[1, 1].set_ylabel("Value")
axes[1, 1].tick_params(axis="x", rotation=45)
axes[1, 1].grid(axis="y", alpha=0.3)
corr = numeric_df.corr()

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
