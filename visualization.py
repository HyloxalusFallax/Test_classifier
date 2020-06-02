import pandas as pd
import matplotlib.pyplot as plt

history = pd.read_csv('history3.csv')
history_style = pd.read_csv('history_style.csv')
history_aug = pd.read_csv('history_aug.csv')
history_both = pd.read_csv('history_both.csv')

print(history["val_accuracy"])
print(history_style["val_accuracy"])
print(history_aug["val_accuracy"])
print(history_both["val_accuracy"])

# xnew = np.linspace(T.min(), T.max(), 300) 

# spl = make_interp_spline(T, power, k=3)  # type: BSpline



fig = plt.figure()

plt.title("Validation accuracy of different augmentaion methods", loc='left', fontsize=12, fontweight=0, color='black')
plt.xlabel("Epochs")
plt.ylabel("Validation accuracy")

plt.plot(history["val_accuracy"].rolling(30).mean(), marker='', color='blue', linewidth=1, alpha=0.9, label='without augumentaion')
plt.plot(history_aug["val_accuracy"].rolling(30).mean(), marker='', color='green', linewidth=1, alpha=0.9, label='traditional augumentaion')
plt.plot(history_style["val_accuracy"].rolling(30).mean(), marker='', color='orange', linewidth=1, alpha=0.9, label='style randomization augmentation')
plt.plot(history_both["val_accuracy"].rolling(30).mean(), marker='', color='red', linewidth=1, alpha=0.9, label='both methods of augumentaion')
plt.legend(loc=2, ncol=1)
plt.show()