import pandas as pd
import matplotlib.pyplot as plt

# hustory for classifier with standard task
history = pd.read_csv('history.csv')
history_style = pd.read_csv('history_style.csv')
history_aug = pd.read_csv('history_aug.csv')
history_both = pd.read_csv('history_both.csv')

print(history["val_accuracy"])
print(history_style["val_accuracy"])
print(history_aug["val_accuracy"])
print(history_both["val_accuracy"])

fig = plt.figure()

plt.title("Validation accuracy of different augmentaion methods", loc='left', fontsize=12, fontweight=0, color='black')
plt.xlabel("Epochs")
plt.ylabel("Validation accuracy")

#using mean over 30 epochs for smoother curves
plt.plot(history["val_accuracy"].rolling(30).mean(), marker='', color='blue', linewidth=1, alpha=0.9, label='without augumentaion')
plt.plot(history_aug["val_accuracy"].rolling(30).mean(), marker='', color='green', linewidth=1, alpha=0.9, label='traditional augumentaion')
plt.plot(history_style["val_accuracy"].rolling(30).mean(), marker='', color='orange', linewidth=1, alpha=0.9, label='style randomization augmentation')
plt.plot(history_both["val_accuracy"].rolling(30).mean(), marker='', color='red', linewidth=1, alpha=0.9, label='both methods of augumentaion')
plt.legend(loc=2, ncol=1)
plt.show()
plt.pause(10)

#section for classifier with dataset shift task

history = pd.read_csv('history_dataset_shift.csv')
history_style = pd.read_csv('history_dataset_shift_style.csv')
history_aug = pd.read_csv('history_dataset_shift_aug.csv')
history_both = pd.read_csv('history_dataset_shift_both.csv')

fig = plt.figure()

plt.title("Validation accuracy of different augmentaion methods", loc='left', fontsize=12, fontweight=0, color='black')
plt.xlabel("Epochs")
plt.ylabel("Validation accuracy")

#using mean over 30 epochs for smoother curves
plt.plot(history["val_accuracy"].rolling(30).mean(), marker='', color='blue', linewidth=1, alpha=0.9, label='without augumentaion')
plt.plot(history_aug["val_accuracy"].rolling(30).mean(), marker='', color='green', linewidth=1, alpha=0.9, label='traditional augumentaion')
plt.plot(history_style["val_accuracy"].rolling(30).mean(), marker='', color='orange', linewidth=1, alpha=0.9, label='style randomization augmentation')
plt.plot(history_both["val_accuracy"].rolling(30).mean(), marker='', color='red', linewidth=1, alpha=0.9, label='both methods of augumentaion')
plt.legend(loc=2, ncol=1)
plt.show()