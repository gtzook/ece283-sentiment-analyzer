import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

labels = ["asserted", "hedged", "speculative"]

cm = np.array([
    [49,  0,  1],
    [45, 41, 14],
    [67, 11, 22],
])

fig, ax = plt.subplots(figsize=(5, 4))

im = ax.imshow(cm, cmap="Blues")

ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=30, ha="right")
ax.set_yticklabels(labels)
ax.set_xlabel("LLM majority vote")
ax.set_ylabel("Rule label")
ax.set_title(r"Rule vs. LLM agreement  ($\kappa = 0.243$, fair)")

thresh = cm.max() / 2
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black", fontsize=11)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig("confusion_matrix.pdf", dpi=150)
fig.savefig("confusion_matrix.png", dpi=150)
print("Saved confusion_matrix.pdf and confusion_matrix.png")
