from datetime import datetime
import csv


# Change at some point to log more generally
def log_training_csv(path: str, batch_size: int, epochs:int, l_rate: float, w_decay: float, acc: float, best: bool, time_taken: float):
        '''
        Log learning rate, weight_decay factor, accuracy, whether network is best yet, to .csv
        '''

        # Write accuracy and time in file:
        with open(path, 'a', newline='') as file:

            timestamp = datetime.now().isoformat(timespec='seconds')

            writer = csv.writer(file, delimiter=',')
            writer.writerow([timestamp, time_taken, batch_size, epochs, l_rate, w_decay, acc, best])



