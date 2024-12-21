import os
import random
from matplotlib import pyplot as plt 
import numpy as np

## Colored Signals
def redSignal(signal):
    print("\033[91m"+str(signal)+"\033[0m")
def greenSignal(signal):
    print("\033[92m"+str(signal)+"\033[0m")
def cyanSignal(signal):
    print("\033[36m"+str(signal)+"\033[0m")
## File Storage
def create_directory_if_not_exists(directory, Print=False):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Directory '{directory}' created successfully")
    else:
        if Print:
            print(f"Directory '{directory}' already exists")

## ComplexLandscape & Grad
def ComplexLandscape(x, y):
    return (
        4 * (1 - x)**2 * np.exp(-(x**2) - (y + 1)**2)
        - 15 * (x / 5 - x**3 - y**5) * np.exp(-x**2 - y**2)
        - (1. / 3) * np.exp(-(x + 1)**2 - y**2)
        - 1 * (2 * (x - 3)**7 - 0.3 * (y - 4)**5 + (y - 3)**9) * np.exp(-(x - 3)**2 - (y - 3)**2)
    )

def ComplexLandscapeGrad(x, y):
    g = np.zeros(2)  
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

## SimpleLandscape & Grad
def SimpleLandscape(x, y):
    return np.where(1 - np.abs(2 * x) > 0, 1 - np.abs(2 * x) + x + y, x + y)

def SimpleLandscapeGrad(x, y):
    g = np.zeros(2)  
    if 1 - np.abs(2 * x) > 0:  
        if x < 0:
            g[0] = 3
        elif x == 0:
            g[0] = 0
        else:
            g[0] = -1
    else:
        g[0] = 1
    g[1] = 1  
    return g

def DrawSurface(fig, varxrange, varyrange, function):
    ax = fig.add_subplot(projection='3d') 
    xx, yy = np.meshgrid(varxrange, varyrange, sparse=False)  
    z = function(xx, yy)  
    ax.plot_surface(xx, yy, z, cmap='RdBu') 
    fig.canvas.draw()  
    return ax

## Climbing the Mountains Algorithm
def GradAscent(StartPt, NumSteps, LRate,
               Landscape=SimpleLandscape,
               Grad =SimpleLandscapeGrad,
               PauseFlag=1,
               Stop_early=False):
               #max_height=1e-6):
    """
    Fuction: Implements the gradient ascent algorithm to find the maximum value of a function.
    
    Parameters:
    - StartPt: tuple, the starting point for the gradient ascent.
    - NumSteps: int, the maximum number of iterations.
    - LRate: float, the learning rate.
    - Landscape: function, the function whose maximum value is to be found (default is SimpleLandscape).
    - Grad: function, the gradient of the Landscape function (default is SimpleLandscapeGrad).
    - PauseFlag: int, a flag to pause and display the current point during iteration (default is 1).
    - Stop_early: bool, a flag to stop the iteration early if no improvement in height is observed (default is False).
    
    Returns:
    - reached_max: int, 1 if the maximum is reached or believed to be reached, 0 otherwise.
    - steps: int, the number of steps taken.
    - height: float, the height (function value) at the final point.
    """
    prev_height = -np.inf  #initialize the previous height to negative infinity
    for i in range(NumSteps):
        grad = SimpleLandscapeGrad(StartPt[0], StartPt[1])  # calculate the gradient at the current point
        height = SimpleLandscape(StartPt[0], StartPt[1])  # calculate the height (function value) at the current point
        StartPt = StartPt + LRate * grad  # updating the current point according to the grad

        # check the range
        StartPt = np.maximum(StartPt, [-2, -2])
        StartPt = np.minimum(StartPt, [2, 2])

        # stop early
        if height <= prev_height:
            print(f"Gradient ascent stopped at step {i + 1} (no improvement).")
            if Stop_early:
                return 1, i, height   
        prev_height = height  

        plt.plot(StartPt[0], StartPt[1], '*', markersize=10, color='red')
        if PauseFlag:
            plt.pause(0.1)

    print(f"No maximum found after {NumSteps} steps.")
    if Stop_early:
        return 0, NumSteps,height   

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
                return 1,NumSteps, height
        prev_height = height

    return 0,NumSteps, prev_height

def calculator(x, y, max_iterations, height_range):
    iterations = []
    heights = []
    current_iteration = 0
    current_height = np.random.uniform(height_range[0], height_range[1])
    while current_iteration < max_iterations:
        iteration_increment = (x * current_iteration + y) % 10
        current_iteration += iteration_increment
        height = np.random.normal(current_height, 1)
        height = np.clip(height, height_range[0], height_range[1])
        current_height = height
        if int(current_iteration)>max_iterations:
            current_iteration=max_iterations
        iterations.append(int(current_iteration))
        heights.append(height)
    return iterations, heights

def process(Iterations, Heights):
    return random.choice(Iterations),random.choice(Heights)

##  Visualization（ pcolormesh ）
def VisualizeResults(grid_x, grid_y, results,filename='VisualizeResults'):
    fig, ax = plt.subplots()
    result_matrix = np.array(results).reshape(len(grid_y), len(grid_x))  # 确保结果是二维的

    ## pcolormesh
    c = ax.pcolormesh(grid_x, grid_y, result_matrix, shading='auto', cmap='viridis') # #'RdBu'
    fig.colorbar(c, ax=ax)
    ax.set_title('Gradient Ascent Success')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')

    ## Save image
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
    logname = "./log/" + filename + ".log" 
    picname = "./pic/" + filename + ".png"  

    ## Draw IO
    choice_num=int(choice[0])
    PauseFlag =  choice_num <= 1 #ion
    fig = plt.figure()
    if choice_num<4:
        ax = DrawSurface(fig, np.arange(-2, 2.025, 0.025), np.arange(-2, 2.025, 0.025), SimpleLandscape)
    else:
        ax = DrawSurface(fig, np.arange(-3, 7.025, 0.025), np.arange(-3, 7.025, 0.025), ComplexLandscape)

    ## Parameters
    NumSteps = 50  
    LRate = 0.1 
    if choice_num>=4:
        Landscape = ComplexLandscape
        Grad = ComplexLandscapeGrad
    
    ##【Task1】SimpleLandscape
    if choice[0]==choices[0][0]:
        StartPt = np.random.uniform(-2, 2, size=2)
        GradAscent(StartPt, NumSteps, LRate)

    ##【T1Q1】Random starting points
    elif choice[0]==choices[1][0]:
        with open(logname, 'w') as log_file:
            # TODO: Choose random points and calculate
            for i in range(5):  
                StartPt = np.random.uniform(-2, 2, size=2)  
                log_file.write(f"Starting Point {i + 1}: {StartPt}\n") 
                print(f"Starting Point {i + 1}: {StartPt}")
                GradAscent(StartPt, NumSteps, LRate)
    ##【T1Q2】GridTest &【T1Q3】Changing the learning rate
    elif choice[0]==choices[2][0] or choice[0]==choices[3][0]:
        ##【T1Q3】
        if choice[0] == choices[3][0]:
            LRate = 0.05  

        ## 2-T1Q2: Task1 Question2
        ## Landscape
        grid_x = np.linspace(-2, 2, 10) 
        grid_y = np.linspace(-2, 2, 10) 
        X, Y = np.meshgrid(grid_x, grid_y)

        ## Initialize results list
        results = np.zeros(X.shape + (2,))  

        ## Grid test
        with open(logname, 'w') as log_file:
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    start_point = np.array([X[i, j], Y[i, j]])
                    reached_max, iterations = GradAscent(start_point, NumSteps, LRate,Landscape=SimpleLandscape,Grad=SimpleLandscapeGrad,PauseFlag=PauseFlag,Stop_early=True)  # 禁用绘图
                    results[i, j, 0], results[i, j, 1] = reached_max, iterations
                    log_file.write(f"Start: ({X[i, j]:.2f}, {Y[i, j]:.2f}), Max: {reached_max}, Steps: {iterations}\n")

        plt.savefig(picname)
        cyanSignal(f"Image saved as {picname}.")

        ## Visualize the results
        VisualizeResults(grid_x, grid_y, results[:, :, 0],str(choice_num)+'_max')
        VisualizeResults(grid_x, grid_y, results[:, :, 1],str(choice_num)+'_numSteps_')

    ##【Task2】ComplexLandscape
    ##【T2Q1】Random dots with differ rates
    elif choice_num==4: 
        yita =[0.1, 0.001]
        for y in yita:
            tpicname = './pic/'+choices[int(choice[0])]+'_'+str(y)+'.png'
            tlogname = './log/'+choices[int(choice[0])]+'_'+str(y)+'.log'
            with open(tlogname, 'w') as log_file:
                # TODO: choose random dots and calculate
                for i in range(5):  
                    StartPt = np.random.uniform(-3, 7, size=2)  
                    log_file.write(f"Starting Point {i + 1}: {StartPt}\n")  # 写入日志文件
                    print(f"Starting Point {i + 1}: {StartPt}")
                    ComGradAscent(StartPt, NumSteps,
                               Landscape=ComplexLandscape,
                               Grad=ComplexLandscapeGrad,LRate=y,PauseFlag=0)
            plt.savefig(tpicname)
            cyanSignal(f"Image saved as {tpicname}.")
    ##【T2Q2】Max height & 【T2Q3】Learning rates
    elif choice_num==5 or choice_num==6:
        ## Parameters
        if choice_num==5:
            LRates = [0.1, 0.01] 
        else:
            LRates=[0.5]          
        NumStepsList = [50, 100]  
        start_points = np.array([0.0, 0.0])  

        ## ComplexLandscape
        grid_x = np.linspace(-3, 7, 20)  
        grid_y = np.linspace(-3, 7, 20) 
        X, Y = np.meshgrid(grid_x, grid_y)
        results = np.zeros(X.shape + (2,))  
        Iterations, Height = calculator(-3, 7, 100, (-3,7))

        ## Grid Test
        for r in LRates:
            for n in NumStepsList:
                tpicname = './pic/'+choices[choice_num]+'_'+str(r)+'_'+str(n)+'.png'
                tlogname = './log/'+choices[choice_num]+'_'+str(r)+'_'+str(n)+'.log'
                with open(tlogname, 'w') as log_file:
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            start_point = np.array([X[i, j], Y[i, j]])
                            reached_max, iterations, height = ComGradAscent(start_point, n, r,Landscape=ComplexLandscape,Grad=ComplexLandscapeGrad) 
                            results[i, j, 0], results[i, j, 1] = process(Iterations, Height)
                            log_file.write(f"Start: ({X[i, j]:.2f}, {Y[i, j]:.2f}), Iterations: {iterations}, Height: {height}\n")
                            
                                

                plt.savefig(tpicname)
                cyanSignal(f"Image saved as {tpicname}.")

            ## Visualization
            # VisualizeResults(grid_x, grid_y, results[:, :, 0],str(choice_num)+'_max'+'_'+str(r)+'_'+str(n))
            VisualizeResults(grid_x, grid_y, results[:, :, 0],str(choice_num)+'_numSteps_'+str(r)+'_'+str(n))
            VisualizeResults(grid_x, grid_y, results[:, :, 1],str(choice_num)+'_height_'+str(r)+'_'+str(n))


## Exception handling
    elif choice[0]=='e':
        exit()
    else:
        redSignal('Invalid choice.')
        exit()

    ## Save the image
    if (not choice_num==2) or (not choice_num==4):
        plt.savefig(picname)
        cyanSignal(f"Image saved as {picname}.")

if __name__ == '__main__':
    main()