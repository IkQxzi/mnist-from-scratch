import numpy as np

from tools.io import read_csv_file


def get_best_net_acc(path: str) -> float:
    '''
    Get the best network accuracy from a .csv file
    
    :param path: .csv file to read from
    :type path: str
    '''

    f_content = read_csv_file(path, delimiter=',')

    # Get the highest accuracy row-by-row
    best_acc: float = 0
    for row in f_content:
        curr_acc = float(row['accuracy'])
        if curr_acc > best_acc:
            best_acc = curr_acc
    
    return best_acc


