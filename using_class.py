import os
import pathlib
import keras.regularizers
import numpy as np
import tensorflow as tf
from keras.layers import Conv2D
import matplotlib.pyplot as plt
import seaborn as sn
import splitfolders

from PIL import Image


def check_image_format(file_path):
    try:
        img = Image.open(file_path)
        img.verify()
        return img.format in ["JPEG", "PNG", "GIF", "BMP"]
    except (IOError, SyntaxError) as e:
        print(f'Bad file: {file_path}')
        return False


def display_samples(dataset, n_samples, classes_name):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(n_samples):
            ax = plt.subplot(5, 5, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(classes_name[np.argmax(labels[i])])
            plt.axis("off")
    plt.show()


def preprocess_image(image_path):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


def clean_directory(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if not check_image_format(file_path):
                os.remove(file_path)
                print(f'Removed: {file_path}')


# Разделение набора данных на тренировочный, валидационный и тестовый наборы
input_folder = "C:\\Users\\user\\Desktop\\dataset"
output_folder = "C:\\Users\\user\\Desktop\\split_dataset"
#splitfolders.ratio(input_folder, output=output_folder, seed=42, ratio=(.7, .2, .1))

# Очистка и проверка изображений в новом наборе данных
clean_directory(output_folder)

dataset_dir = pathlib.Path(output_folder)

batch_size = 32
img_width = 180
img_height = 180

train = tf.keras.utils.image_dataset_from_directory(
    dataset_dir / 'train',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

valid = tf.keras.utils.image_dataset_from_directory(
    dataset_dir / 'val',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

test = tf.keras.utils.image_dataset_from_directory(
    dataset_dir / 'test',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode="categorical"
)

class_names_len = len(train.class_names)
class_names = train.class_names
# display_samples(train, 25, class_names)

AUTOTUNE = tf.data.experimental.AUTOTUNE

train = train.prefetch(buffer_size=AUTOTUNE)
valid = valid.prefetch(buffer_size=AUTOTUNE)
test = test.prefetch(buffer_size=AUTOTUNE)

# Аугментация данных
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
    tf.keras.layers.RandomContrast(0.2),
])

train = train.map(lambda x, y: (data_augmentation(x, training=True), y))

base_model = tf.keras.applications.VGG16(input_shape=(img_height, img_width, 3),
                                         include_top=False,
                                         weights='imagenet')

base_model.trainable = False

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(class_names_len, activation='softmax')
])

model.summary()

model.compile(
    loss="categorical_crossentropy",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    metrics=["accuracy"]
)


# Оценка модели на тестовых данных
scores = model.evaluate(test, verbose=1)
print("Точность тестирования", round(scores[1] * 100, 4))
