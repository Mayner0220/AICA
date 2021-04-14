import os
import matplotlib.pyplot as plt
from data_load import train_ds

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(train_ds.class_names[labels[i]])
        plt.axis("off")

img_save_path = "./img/"
try:
    if not(os.path.isdir(img_save_path)):
        os.makedirs(os.path.join(img_save_path))

    plt.savefig(img_save_path + "train_data_visualize.png", facecolor="#eeeeee", bbox_inches="tight")
    plt.show()
except:
    print("[Failed to create directory]")