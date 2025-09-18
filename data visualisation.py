import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(r"C:\Users\10108\Desktop\ncr_ride_bookings.csv")
print(df.head())
# 设置最多显示 200 行
pd.set_option("display.max_rows", 200)
df.to_csv("output.csv", index=False, encoding="utf-8-sig")
# 查看列名
#print(df.columns)

# 基本统计信息
print(df.describe())
# 取 describe() 结果
desc = df.describe()

# 只画均值
desc.loc["mean"].plot(kind="bar", figsize=(10, 6), alpha=0.7)

plt.title("Average Values of Numerical Features", fontsize=16, fontweight="bold")
plt.ylabel("Mean Value")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.show()

#箱线图
df[["Avg VTAT", "Avg CTAT", "Driver Ratings", "Customer Rating"]].plot(
    kind="box", figsize=(8, 6)
)
plt.title("Boxplot of Key Metrics", fontsize=16, fontweight="bold")
plt.ylabel("Value")
plt.show()

df[["Avg VTAT", "Avg CTAT", "Driver Ratings", "Customer Rating"]].hist(
    bins=20, figsize=(10, 8), grid=False
)
plt.suptitle("Distribution of Features", fontsize=16, fontweight="bold")
plt.show()

stats = desc.loc[["mean", "50%"]]  # mean vs median
stats.T.plot(kind="bar", figsize=(10, 6))

plt.title("Comparison of Mean and Median", fontsize=16, fontweight="bold")
plt.ylabel("Value")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.show()
