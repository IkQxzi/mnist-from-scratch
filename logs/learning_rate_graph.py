import matplotlib.pyplot as plt


def read_learning_rates() -> tuple[list[float], list[float]]:

    with open("results.txt", "r") as f:
        lines = f.readlines()

    # [time] learning_rate: <value>, accuracy: <value>

    learning_rates = []
    accs = []

    for line in lines:
        learning_rate = float(line.split(",")[0].split(": ")[1])
        learning_rate = round(learning_rate, 3)  # To remove floating-point errors
        learning_rates.append(learning_rate)

        acc = float(line.split("accuracy: ")[1].strip("\n"))
        accs.append(acc)

    return learning_rates, accs


def plot_points(x_points: list, y_points: list) -> None:

    plt.plot(x_points, y_points, "o")
    plt.show()


learning_rates, accs = read_learning_rates()
plot_points(x_points=learning_rates, y_points=accs)
