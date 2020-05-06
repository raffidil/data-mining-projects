import numpy as np
header_sting = """relation exuation
attribute xone real
attribute xtwo real
attribute output {-1,1}
data"""
file_path = '/home/mohammad/GIT projects/GitHub/data-mining-projects/weka_practice/dataset.arff'
dataset = np.loadtxt(file_path, delimiter = ',', comments='@' )
different_sizes = [50, 100, 250, 1650, 2550]
for size in different_sizes:
    index_array = np.random.randint(low = 0,high = 6000,size = size , dtype="int16")
    chosen_data = dataset[index_array,:]
    np.savetxt('/home/mohammad/GIT projects/GitHub/data-mining-projects/weka_practice/randomlyChosen%s'%str(size)+'.arff',
                chosen_data,delimiter=',',header=header_sting, comments='@')