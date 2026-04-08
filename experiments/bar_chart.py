import matplotlib.pyplot as plt

conf_lbls = [i for i in range(10)]
conf_scores = [i/10 for i in conf_lbls]

plt.bar(x=conf_lbls, height=conf_scores)
plt.show()