import os
import pathlib
import matplotlib.pyplot as plt
from keras_preprocessing.image import ImageDataGenerator
from lib.helper import Helper

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = Helper.load_data(
        os.path.join(pathlib.Path(__file__).parent.parent.absolute(), "lib/mnist.npz")
    )
    aug = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1)
    image = train_images[300]
    plt.imshow(image)
    plt.show()
    m = image.reshape([-1, 28, 28, 1])
    imageGen = aug.flow(m, batch_size=1)

    images = []
    total = 0
    for img in imageGen:
        total += 1
        images.append(img)
        if total == 5:
            break

    print(len(images))

    for n in images:
        plt.imshow(n.reshape([28, 28]))
        plt.show()
