import os
import random

import matplotlib.pyplot as plt
from PIL import Image

def main():
    dataset_path = "../../DATA/clouds/clouds_train"

    categories = {}
    for category in os.listdir(dataset_path):
        category_path = os.path.join(dataset_path, category)
        if os.path.isdir(category_path):
            categories[category] = [os.path.join(category_path, img) for img in os.listdir(category_path) if img.endswith(".jpg")]

    all_images = [(img, category) for category, images in categories.items() for img in images]
    selected_images = random.sample(all_images, 6)

    fig, axes = plt.subplots(2, 3)
    fig.subplots_adjust(hspace=0.5)

    for ax, (img_path, category) in zip(axes.flat, selected_images):
        ax.imshow(Image.open(img_path))
        ax.axis("off")
        ax.set_title(category)

    fig.suptitle("The Clouds dataset")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
