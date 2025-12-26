import matplotlib.pyplot as plt


def read_learning_rates() -> tuple[list[float], list[float]]:
    
    with open('results.txt', 'r') as f:
        lines = f.readlines()

    # [time] learning_rate: <value>, accuracy: <value>
    
    l_rates = []
    accs = []

    for line in lines: 
        l_rate = float(line.split(',')[0].split(': ')[1])
        l_rate = round(l_rate, 3) # To remove floating-point errors
        l_rates.append(l_rate)

        acc = float(line.split('accuracy: ')[1].strip('\n'))
        accs.append(acc)

    return l_rates, accs

def plot_points(x_points: list, y_points: list) -> None:

    plt.plot(x_points, y_points, 'o')
    plt.show()


l_rates, accs = read_learning_rates()
plot_points(x_points=l_rates, y_points=accs)

    

