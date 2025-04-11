# import matplotlib.pyplot as plt
#
# x = [1, 2, 3, 4, 5]  # 确定柱状图数量，可以认为是x方向刻度
# y = [1118, 666, 851, 1097, 391]  # y方向刻度
#
#
# x_label = ['good', 'broke', 'lose', 'uncovered', 'circle']
# plt.xticks(x, x_label)  # 绘制x刻度标签
# bars = plt.bar(x, y, color='#00a2ff')  # 绘制y刻度标签，并设置颜色为蓝色
#
# # 在每个柱状图的上方标注数量
# for bar, value in zip(bars, y):
#     plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(value), ha='center', va='bottom')
#
# # 去掉左右两边的竖着的y轴刻度线
# plt.gca().spines['left'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# plt.gca().spines['top'].set_visible(False)
#
# # 在y轴上添加刻度线200、400、600、800、1000
# plt.yticks([200, 400, 600, 800, 1000])
#
# # 设置网格刻度
# plt.grid(axis='y', linestyle='-', alpha=0.7)
#
# # 显示图形
# plt.show()

import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]  # 确定柱状图数量，可以认为是x方向刻度
y = [1238, 1233, 1249, 1012, 1188]  # y方向刻度


x_label = ['potholes', 'cracks', 'subsidence', 'rut', 'chap']
plt.xticks(x, x_label)  # 绘制x刻度标签
bars = plt.bar(x, y, color='#00a2ff')  # 绘制y刻度标签，并设置颜色为蓝色

# 在每个柱状图的上方标注数量
for bar, value in zip(bars, y):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10, str(value), ha='center', va='bottom')

# 去掉左右两边的竖着的y轴刻度线
plt.gca().spines['left'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

# 在y轴上添加刻度线200、400、600、800、1000
plt.yticks([200, 400, 600,800,1000,1200,1400])
# plt.yticks([200, 400, 600])
# 设置网格刻度
plt.grid(axis='y', linestyle='-', alpha=0.7)

# 显示图形
plt.show()
