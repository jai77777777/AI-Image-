import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

IMG_SIZE = (160, 160)
BATCH_SIZE = 64
EPOCHS = 3


datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)
print(train_data.class_indices)


val_data = datagen.flow_from_directory(
    "dataset",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(160, 160, 3)
)


base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS
)
print(train_data.class_indices)

model.save("ai_image_detector.h5")
print("✅ Model saved as ai_image_detector.h5")
