import zipfile
import numpy as np
from PIL import Image
import os
import glob

import tensorflow as tf
from tensorflow import keras
import tensorflow.lite as tflite

from tensorflow.keras.preprocessing.image import load_img, ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB2
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint

# Data preparation

zip_file = "rice_leaf_datasets.zip"
extract_to = "rice_leaf_datasets"

try:
    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted files to: {extract_to}")
except zipfile.BadZipFile:
    print("Error: File is not a valid ZIP archive or is corrupted.")

path = "./rice_leaf_datasets/train/Blast"
name = "BLAST1_008.jpg"
img_name = f"{path}/{name}"

img = load_img(img_name, target_size=(299,299))
img_array = np.array(img)


# Model

# * 1. Build and train CNN base model

train_gen_efficientnetb2 = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)
train_dataset = train_gen_efficientnetb2.flow_from_directory("./rice_leaf_datasets/train", target_size=(260,260), batch_size=32)

# X, y = next(train_dataset)

valid_gen_efficientnetb2 = ImageDataGenerator(preprocessing_function=tf.keras.applications.efficientnet.preprocess_input)
valid_dataset = valid_gen_efficientnetb2.flow_from_directory("./rice_leaf_datasets/val", target_size=(260,260), batch_size=32, shuffle=False)

def make_efficientnetb2_model(learning_rate=0.01, size_inner=0, droprate=0.0):
    base_model = EfficientNetB2(weights="imagenet", include_top=False, input_shape=(260, 260, 3))
    base_model.trainable = False


    inputs = keras.Input(shape=(260,260,3))
    base = base_model(inputs, training=False) 
    vectors = keras.layers.GlobalAveragePooling2D()(base) 

    inner = keras.layers.Dense(size_inner, activation="relu")(vectors)
    dropout = keras.layers.Dropout(droprate)(inner)
    
    outputs = keras.layers.Dense(4)(dropout) 
    model = keras.Model(inputs, outputs)


    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    loss = keras.losses.CategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model

# * 2. Create final model

efficientnetb2_checkpoint = ModelCheckpoint("efficientnetb2_{epoch:02d}_{val_accuracy:.3f}.keras", save_best_only=True, monitor="val_accuracy", mode="max")

efficientnetb2_model = make_efficientnetb2_model(learning_rate=0.001, size_inner=1000, droprate=0.0)

history = efficientnetb2_model.fit(train_dataset, epochs=5, validation_data=valid_dataset, callbacks=[efficientnetb2_checkpoint])


# Using model

files = glob.glob("efficientnetb2_*.keras")  # Get all files matching the pattern
if files:
    # Extract accuracy and epoch from filenames
    file_with_max_accuracy = max(
        files,
        key=lambda f: float(f.split("_")[2].replace(".keras", ""))
    )
    new_filename = "rice-leaf-disease-prediction-model.keras"
    
    # Rename the file with maximum accuracy
    os.rename(file_with_max_accuracy, new_filename)
    print(f"File renamed to {new_filename}")
else:
    print("No model files found!")

final_model = keras.models.load_model("rice-leaf-disease-prediction-model.keras")

test_gen_efficientnet = ImageDataGenerator(preprocessing_function=preprocess_input)
test_dataset = test_gen_efficientnet.flow_from_directory("./rice_leaf_datasets/test", target_size=(260,260), batch_size=32, shuffle=False)

final_model.evaluate(test_dataset)

path = "./rice_leaf_datasets/test/Tungro"
name = "TUNGRO1_036.jpg"
img_name = f"{path}/{name}"
img = load_img(img_name, target_size=(260,260))

img_array = np.array(img, dtype="float32")
img = np.array([img_array])
img = preprocess_input(img)

pred = final_model.predict(img)

classes = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

print(("The output of first image: \n") + str(dict(zip(classes, pred[0]))))


# Convert keras model to tf-lite and save the model

model_converter = tf.lite.TFLiteConverter.from_keras_model(final_model)
tflite_model = model_converter.convert()

with open ("rice-leaf-disease-prediction-model.tflite", "wb") as f_out:
    f_out.write(tflite_model)

model_interpreter = tflite.Interpreter(model_path="rice-leaf-disease-prediction-model.tflite")
model_interpreter.allocate_tensors()

input_index = model_interpreter.get_input_details()[0]['index']
output_index = model_interpreter.get_output_details()[0]['index']

model_interpreter.set_tensor(input_index, img)
model_interpreter.invoke()
preds = model_interpreter.get_tensor(output_index)


with Image.open("./rice_leaf_datasets/val/Bacterialblight/BACTERAILBLIGHT5_037.jpg") as img:
    img = img.resize((260, 260), Image.NEAREST)

img_array = np.array(img, dtype="float32")
img = np.array([img_array])
img = preprocess_input(img)

model_interpreter.set_tensor(input_index, img)
model_interpreter.invoke()
preds = model_interpreter.get_tensor(output_index)

classes = ['Bacterialblight', 'Blast', 'Brownspot', 'Tungro']

print(("The output of second image: \n") + str(dict(zip(classes, pred[0]))))