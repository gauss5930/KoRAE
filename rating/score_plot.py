from datasets import load_dataset
import matplotlib.pyplot as plt

rated_KoRAE = load_dataset("Cartinoe5930/KoRAE_rated", split="train")

score_result = {}
x_label = []
for i in range(0, 21):
    score_result[i * 0.5] = 0
    x_label.append(i * 0.5)

for data in rated_KoRAE:
    score_result[float(data["score"])] += 1

x = list(score_result.keys())
y = list(score_result.values())

plt.figure(figsize=(10, 6))
plt.bar(x, y, width=0.4, align='center', alpha=0.7)
plt.xlabel('Scores')
plt.ylabel('Frequency')
plt.title('Score Distribution Plot')
plt.xticks(x, rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.show()