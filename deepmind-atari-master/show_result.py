import os
import numpy as np
import matplotlib.pyplot as plt

score_file = open('./dqn/score.dat')
eval_file = open('./dqn/eval.dat')

score = []
times = []

for line in score_file:
	score.append(line)

for line in eval_file:
	times.append(line)

y = np.array(score)
x = np.array(times)

plt.plot(x,y,color="blue")
plt.xlabel('evaluation_times')
plt.ylabel('average_score')
plt.title("Learning Process")
plt.legend()
plt.show()

score_file.close()
eval_file.close()

