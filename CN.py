# 导入必要的模块
import os
from matplotlib import pyplot as plt  # 绘图库
import numpy as np  # 数值计算库

#Colored Signals
def redSignal(signal):
    print("\033[91m"+str(signal)+"\033[0m")
def greenSignal(signal):
    print("\033[92m"+str(signal)+"\033[0m")
def cyanSignal(signal):
    print("\033[36m"+str(signal)+"\033[0m")
#File Storage
def create_directory_if_not_exists(directory, Print=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully")
    else:
        if Print:
            print(f"Directory '{directory}' already exists")

# 定义复杂地形函数
def ComplexLandscape(x, y):
    return (
        4 * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2)
        - 15 * (x / 5 - x**3 - y**5) * np.exp(-x**2 - y**2)
        - (1. / 3) * np.exp(-(x + 1)**2 - y**2)
        - 1 * (2 * (x - 3)**7 - 0.3 * (y - 4)**5 + (y - 3)**9) * np.exp(-(x - 3)**2 - (y - 3)**2)
    )

# 定义复杂地形的梯度
def ComplexLandscapeGrad(x, y):
    g = np.zeros(2)  # 初始化梯度为零向量
    g[0] = (
        -8 * np.exp(-(x**2) - (y + 1)**2) * ((1 - x) + x * (1 - x)**2)
        - 15 * np.exp(-x**2 - y**2) * ((0.2 - 3 * x**2) - 2 * x * (x / 5 - x**3 - y**5))
        + (2. / 3) * (x + 1) * np.exp(-(x + 1)**2 - y**2)
        - 1 * np.exp(-(x - 3)**2 - (y - 3)**2) * (14 * (x - 3)**6 - 2 * (x - 3) * (2 * (x - 3)**7 - 0.3 * (y - 4)**5 + (y - 3)**9))
    )
    g[1] = (
        -8 * (y + 1) * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2)
        - 15 * np.exp(-x**2 - y**2) * (-5 * y**4 - 2 * y * (x / 5 - x**3 - y**5))
        + (2. / 3) * y * np.exp(-(x + 1)**2 - y**2)
        - 1 * np.exp(-(x - 3)**2 - (y - 3)**2) * ((-1.5 * (y - 4)**4 + 9 * (y - 3)**8) - 2 * (y - 3) * (2 * (x - 3)**7 - 0.3 * (y - 4)**5 + (y - 3)**9))
    )
    return g

# 定义简单地形函数
def SimpleLandscape(x, y):
    return np.where(1 - np.abs(2 * x) > 0, 1 - np.abs(2 * x) + x + y, x + y)

# 定义简单地形的梯度
def SimpleLandscapeGrad(x, y):
    g = np.zeros(2)  # 初始化梯度为零向量
    if 1 - np.abs(2 * x) > 0:  # 判断是否在有效区域
        if x < 0:
            g[0] = 3
        elif x == 0:
            g[0] = 0
        else:
            g[0] = -1
    else:
        g[0] = 1
    g[1] = 1  # y方向的梯度始终为1
    return g

# 绘制地形表面函数
def DrawSurface(fig, varxrange, varyrange, function):
    """
    绘制给定函数的表面图。
    参数：
    - fig: Matplotlib 图形对象
    - varxrange: x 轴范围
    - varyrange: y 轴范围
    - function: 要绘制的函数
    """
    ax = fig.add_subplot(projection='3d')  # 创建3D绘图对象
    xx, yy = np.meshgrid(varxrange, varyrange, sparse=False)  # 创建网格
    z = function(xx, yy)  # 计算函数值
    ax.plot_surface(xx, yy, z, cmap='RdBu')  # 绘制表面，使用红蓝色彩映射
    fig.canvas.draw()  # 更新绘图
    return ax

# 梯度上升算法
def GradAscent(StartPt, NumSteps, LRate,
               Landscape=SimpleLandscape,
               Grad =SimpleLandscapeGrad,
               PauseFlag=1,
               Stop_early=False):
               #max_height=1e-6):
    """
    实现梯度上升算法，找到函数的最大值。
    参数：
    - StartPt: 起始点
    - NumSteps: 最大迭代次数
    - LRate: 学习率
    - Landscape: SimpleLandscape,ComplexLandscape
    - PauseFlag: 默认为 1，暂停标志，用于交互式显示
    - max_height: 梯度收敛判定的容忍度
    返回：
    - reached_max: 1是
    - steps: 最后的步数
    """
    prev_height = -np.inf  # 初始设定为负无穷，确保第一次计算时一定会更新
    for i in range(NumSteps):
        grad = SimpleLandscapeGrad(StartPt[0], StartPt[1])  # 计算当前点的梯度
        height = SimpleLandscape(StartPt[0], StartPt[1])  # 计算当前点的高度
        StartPt = StartPt + LRate * grad  # 根据梯度更新当前点

        # 确保当前点在指定范围内
        StartPt = np.maximum(StartPt, [-2, -2])
        StartPt = np.minimum(StartPt, [2, 2])

        # 如果当前高度没有比前一步高，认为达到最大值，停止梯度上升
        if height <= prev_height:
            print(f"Gradient ascent stopped at step {i + 1} (no improvement).")
            if Stop_early:
                return 1, i  # 返回1表示找到了最大值，i表示步数

        # if height >= max_height:
        #     return 1, i

        prev_height = height  # 更新前一步的高度

        # 在地形图上绘制当前点
        plt.plot(StartPt[0], StartPt[1], '*', markersize=10, color='red')
        # 暂停以查看当前点
        if PauseFlag:
            plt.pause(0.1)

    print(f"No maximum found after {NumSteps} steps.")
    if Stop_early:
        return 0, NumSteps  # 返回0表示未找到最大值，NumSteps表示总步数


def ComGradAscent(StartPt, NumSteps, LRate, Landscape, Grad, PauseFlag=0, Stop_early=False):
    prev_height = -np.inf
    for i in range(NumSteps):
        grad = ComplexLandscapeGrad(StartPt[0], StartPt[1])
        height = ComplexLandscape(StartPt[0], StartPt[1])
        StartPt = StartPt + LRate * grad
        StartPt = np.maximum(StartPt, [-3, 7])
        StartPt = np.minimum(StartPt, [-3, 7])
        if height <= prev_height:
            if Stop_early:
                return 1, height  # 返回1表示停止，返回当前的高度
        prev_height = height

    return 0, prev_height


def ComGradAscent_(StartPt, NumSteps, LRate, Landscape, Grad, PauseFlag=1, Stop_early=False):
    prev_height = -np.inf
    for i in range(NumSteps):
        grad = Grad(StartPt[0], StartPt[1])
        height = Landscape(StartPt[0], StartPt[1])
        StartPt = StartPt + LRate * grad
        StartPt = np.maximum(StartPt, [-3, 7])
        StartPt = np.minimum(StartPt, [-3, 7])
        if height <= prev_height:
            if Stop_early:
                return 1, height  # 返回1表示停止，返回当前的高度
        prev_height = height

    return 0, prev_height
# pcolormesh可视化
def VisualizeResults(grid_x, grid_y, results,filename='VisualizeResults'):
    fig, ax = plt.subplots()

    # 将 results 转换为一个二维数组，确保它与网格的大小匹配
    result_matrix = np.array(results).reshape(len(grid_y), len(grid_x))  # 确保结果是二维的

    # 使用 pcolormesh 绘制结果
    c = ax.pcolormesh(grid_x, grid_y, result_matrix, shading='auto', cmap='viridis') # #'RdBu'

    # 添加颜色条
    fig.colorbar(c, ax=ax)

    # 设置标题和标签
    ax.set_title('Gradient Ascent Success')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    # 保存图像
    filename='./pic/'+filename+'.png'
    plt.savefig(filename)
    plt.show()

def main():
    ## Choices
    choices=['0-init',
             '1-T1Q1',
             '2-T1Q2',
             '3-T1Q3',
             '4-T2Q1',
             '5-T2Q2',
             '6-T2Q3',
             'e-exit'
            ]
    greenSignal("Input the choice:")
    for c in choices:
        print(c)
    choice = input()
    ## Results storage
    filename = choices[int(choice[0])]
    create_directory_if_not_exists('./pic')
    create_directory_if_not_exists('./log')
    logname = "./log/" + filename + ".log"  ## 定义日志文件名
    picname = "./pic/" + filename + ".png"  ## 定义图像文件名

    ## Draw IO
    # 绘制地形图（根据需要选择简单地形或复杂地形）
    choice_num=int(choice[0])
    PauseFlag =  choice_num <= 1 #ion
    fig = plt.figure()
    if choice_num<4:
        ax = DrawSurface(fig, np.arange(-2, 2.025, 0.025), np.arange(-2, 2.025, 0.025), SimpleLandscape)
    else:
        ax = DrawSurface(fig, np.arange(-3, 7.025, 0.025), np.arange(-3, 7.025, 0.025), ComplexLandscape)

    ## Parameters
    NumSteps = 50  # 最大迭代次数
    LRate = 0.1  # 学习率

    if choice_num>=4:
        Landscape = ComplexLandscape
        Grad = ComplexLandscapeGrad

    #Task1 SimpleLandscape
    if choice[0]==choices[0][0]:
        # 初始化起点，范围为(-2, 2)（简单地形）或(-3, 7)（复杂地形）
        StartPt = np.random.uniform(-2, 2, size=2)

        # 运行梯度上升算法
        GradAscent(StartPt, NumSteps, LRate)
    elif choice[0]==choices[1][0]:
        with open(logname, 'w') as log_file:
            # TODO: 选择多个随机起点，并运行梯度上升算法
            for i in range(5):  # 从5个不同的起点开始
                StartPt = np.random.uniform(-2, 2, size=2)  # 随机生成起点，范围在[-2, 2]
                log_file.write(f"Starting Point {i + 1}: {StartPt}\n")  # 写入日志文件
                print(f"Starting Point {i + 1}: {StartPt}")
                GradAscent(StartPt, NumSteps, LRate)
    elif choice[0]==choices[2][0] or choice[0]==choices[3][0]:
        ## 3-T1Q3: Task1 Question3
        if choice[0] == choices[3][0]:
            LRate = 0.05  # 学习率

        ## 2-T1Q2: Task1 Question2
        ## 定义地形网格
        grid_x = np.linspace(-2, 2, 10)  # 使用10个网格点覆盖x轴
        grid_y = np.linspace(-2, 2, 10)  # 使用10个网格点覆盖y轴
        X, Y = np.meshgrid(grid_x, grid_y)

        ## 初始化结果存储
        results = np.zeros(X.shape + (2,))  # 存储是否到达最大值和迭代次数

        # 使用网格覆盖点进行梯度上升算法测试
        with open(logname, 'w') as log_file:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    start_point = np.array([X[i, j], Y[i, j]])
                    reached_max, iterations = GradAscent(start_point, NumSteps, LRate,Landscape=SimpleLandscape,Grad=SimpleLandscapeGrad,PauseFlag=PauseFlag,Stop_early=True)  # 禁用绘图
                    results[i, j, 0], results[i, j, 1] = reached_max, iterations
                    log_file.write(f"Start: ({X[i, j]:.2f}, {Y[i, j]:.2f}), Max: {reached_max}, Steps: {iterations}\n")

        plt.savefig(picname)
        cyanSignal(f"Image saved as {picname}.")

        # 可视化是否到达最大值的图
        VisualizeResults(grid_x, grid_y, results[:, :, 0],str(choice_num)+'_max')
        # 可视化达到最大值所需的迭代次数图
        VisualizeResults(grid_x, grid_y, results[:, :, 1],str(choice_num)+'_numSteps_')

    #Task2 ComplexLandscape
    elif choice_num==4: #Random dots with differ rates
        yita =[0.1, 0.001]
        for y in yita:
            tpicname = './pic/'+choices[int(choice[0])]+'_'+str(y)+'.png'
            tlogname = './log/'+choices[int(choice[0])]+'_'+str(y)+'.log'
            with open(tlogname, 'w') as log_file:
                # TODO: 选择多个随机起点，并运行梯度上升算法
                for i in range(5):  # 从5个不同的起点开始
                    StartPt = np.random.uniform(-2, 2, size=2)  # 随机生成起点，范围在[-2, 2]
                    log_file.write(f"Starting Point {i + 1}: {StartPt}\n")  # 写入日志文件
                    print(f"Starting Point {i + 1}: {StartPt}")
                    ComGradAscent(StartPt, NumSteps,
                               Landscape=ComplexLandscape,
                               Grad=ComplexLandscapeGrad,LRate=y,PauseFlag=0)
            plt.savefig(tpicname)
            cyanSignal(f"Image saved as {tpicname}.")
    elif choice_num==5 or choice_num==6:

        if choice_num==5:
            LRates = [0.1, 0.01]  # 两种不同的学习率
        else: # choice_num == 6:
                LRates=0.5
        NumStepsList = [50, 100]  # 两种不同的迭代次数
        start_points = np.array([0.0, 0.0])  # 起始点




        results = {}

        # 进行实验并记录每次的高度变化
        for LRate in LRates:
            for NumSteps in NumStepsList:
                heights = []
                for i in range(NumSteps):
                    _, height = ComGradAscent_(start_points, NumSteps, LRate, ComplexLandscape, ComplexLandscapeGrad,
                                              PauseFlag=0, Stop_early=False)
                    heights.append(height)
                results[(LRate, NumSteps)] = heights

        # 转换 heights 为二维数组，确保符合 pcolormesh 的要求
        LRates = np.array(LRates)
        NumStepsList = np.array(NumStepsList)

        plt.figure(figsize=(8, 6))
        for idx, (LRate, NumSteps) in enumerate(results):
            heights = np.array(results[(LRate, NumSteps)])

            # 创建一个 NumSteps x len(LRates) 的二维数组
            heights_2d = np.reshape(heights, (len(LRates), -1))

            # 绘制每个组合的实验结果
            plt.subplot(2, 2, idx + 1)
            plt.pcolormesh(np.arange(NumSteps), LRates, heights_2d, shading='auto')
            plt.colorbar()
            plt.title(f"LRate={LRate}, NumSteps={NumSteps}")
            plt.xlabel("Iteration")
            plt.ylabel("Learning Rate")

        plt.tight_layout()
        plt.show()

## Exception handling
    elif choice[0]=='e':
        exit()
    else:
        redSignal('Invalid choice.')
        exit()

    ## 保存图像
    if (not choice_num==2) or (not choice_num==4):
        plt.savefig(picname)
        cyanSignal(f"Image saved as {picname}.")

if __name__ == '__main__':
    main()