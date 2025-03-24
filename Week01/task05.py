import numpy as np


if __name__ == '__main__':
    baseball = [[180, 78.4],
            [215, 102.7],
            [210, 98.5],
            [188, 75.2]]
    np_baseball = np.array(baseball)
    print("Type:", type(np_baseball))
    print("Number of rows and columns:", np_baseball.shape)
