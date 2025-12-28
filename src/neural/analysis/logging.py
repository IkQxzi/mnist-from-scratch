import time
import csv


# Change at some point to log more generally
def log_training_csv(l_rate: float, w_decay: float, acc: float, best: bool):
        '''
        Log learning rate, weight_decay factor, accuracy, whether network is best yet, to .csv
        '''

        # Write accuracy and time in file:
        with open('logs//test.csv', 'a', newline='') as file:

            time_now = time.strftime('%H:%M:%S')

            writer = csv.writer(file, delimiter=',')
            writer.writerow([l_rate, w_decay, acc, best])



log_training_csv(0.01, 1e-4, 98.97, True)



