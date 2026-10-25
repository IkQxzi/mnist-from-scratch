'''
Contains logging & evaluation functions.
'''

import csv
from datetime import datetime

from tools.io import read_csv_file


def get_best_net_acc(path: str) -> float:
    '''
    Get the best network accuracy from a .csv file
    
    :param path: .csv file to read from
    :type path: str
    '''

    f_content = read_csv_file(path, delimiter=',')

    # Get the highest accuracy by lkooking for the latest 'best' net
    for row in reversed(f_content):
        if eval(row['best']) == True:
            best_acc = float(row['accuracy'])
        
            return best_acc
    
    return 0


# Change at some point to log more generally
def log_training_csv(path: str, batch_size: int, epochs:int, learning_rate: float, weight_decay: float, acc: float, best: bool, time_taken: float):
        '''
        Log learning rate, weight_decay factor, accuracy, whether network is best yet, to .csv
        '''

        # Write accuracy and time in file:
        with open(path, 'a', newline='') as file:

            timestamp = datetime.now().isoformat(timespec='seconds')

            writer = csv.writer(file, delimiter=',')
            writer.writerow([timestamp, time_taken, batch_size, epochs, learning_rate, weight_decay, acc, best])

