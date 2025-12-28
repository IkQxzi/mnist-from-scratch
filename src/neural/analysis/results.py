import numpy as np

from tools.io import read_csv_file


def get_best_net_acc(path: str) -> float:
    '''
    Get the best network accuracy from a .csv file
    
    :param path: .csv file to read from
    :type path: str
    '''

    f_content = read_csv_file(path, delimiter=',')

    # Get the highest accuracy by looking for the latest 'best' net
    for row in reversed(f_content):
        if eval(row['best']) == True:
            best_acc = float(row['accuracy'])
        
            return best_acc
    
    raise IndexError()



