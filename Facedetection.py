
import os
import cv2
import uuid
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Flatten, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.regularizers import l2
from sklearn.metrics import precision_score, recall_score, accuracy_score

print(os.getcwd())

# Define paths
ANC_PATH = os.path.join('data', 'anchor')
POS_PATH = os.path.join('data', 'positive')
NEG_PATH = os.path.join('data', 'negative')


os.makedirs(ANC_PATH, exist_ok=True)
os.makedirs(POS_PATH, exist_ok=True)
os.makedirs(NEG_PATH, exist_ok=True)

cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break


    frame = frame[120:120+250, 200:200+250, :]

    cv2.imshow('Image Collection', frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('a'):  # 'a' for anchor
        imgname = os.path.join(ANC_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
        print(f"Anchor image saved at {imgname}")
    elif key == ord('p'):  # 'p' for positive
        imgname = os.path.join(POS_PATH, '{}.jpg'.format(uuid.uuid1()))
        cv2.imwrite(imgname, frame)
        print(f"Positive image saved at {imgname}")
    elif key == ord('q'):  # 'q' to quit
        print("Exiting data collection...")
        break

cap.release()
cv2.destroyAllWindows()
import tensorflow as tf

# Data Augmentation Function
def data_augmentation(img):
    # Random brightness adjustment
    img = tf.image.random_brightness(img, max_delta=0.2)

    # Random contrast adjustment
    img = tf.image.random_contrast(img, lower=0.5, upper=1.5)

    # Random flipping
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)

    # Random rotation
    img = tf.image.rot90(img, k=tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32))

    # Random zoom (scale)
    img = tf.image.resize_with_crop_or_pad(img, target_height=110, target_width=110)
    img = tf.image.random_crop(img, size=[100, 100, 3])

    return img

# Prepare datasets
anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').shuffle(1000)
positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').shuffle(1000)
negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').shuffle(1000)

# Preprocessing function
def preprocess(file_path,augment=True):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    img = tf.ensure_shape(img, [100, 100, 3])
    if augment:
        img = data_augmentation(img)
    return img


def preprocess_twin(anchor_path, validation_path, label):
    return (preprocess(anchor_path), preprocess(validation_path)), label


positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# Preprocess and prepare the data
data = data.map(preprocess_twin)
data = data.shuffle(1024)
data = data.batch(16)
data = data.prefetch(8)

# Split data into training and validation sets
total_samples = len(list(anchor))
train_samples = int(0.7 * total_samples)
train_data = data.take(train_samples)
val_data = data.skip(train_samples).take(total_samples - train_samples)


def make_embedding():
    return tf.keras.Sequential([
        Conv2D(64, (10, 10), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(2, 2),
        Conv2D(128, (7, 7), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(2, 2),
        Conv2D(128, (4, 4), activation='relu', kernel_regularizer=l2(0.01)),
        MaxPooling2D(2, 2),
        Conv2D(256, (4, 4), activation='relu', kernel_regularizer=l2(0.01)),
        Flatten(),
        Dense(4096, activation='sigmoid', kernel_regularizer=l2(0.01))
    ])

embedding = make_embedding()

# Define L1 Distance Layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)

# Build Siamese Model
def siamese_model():
    input_anchor = Input(name='Anchor', shape=(100, 100, 3))
    input_positive = Input(name='Positive', shape=(100, 100, 3))

    embedding_anchor = embedding(input_anchor)
    embedding_positive = embedding(input_positive)

    l1 = L1Dist()(embedding_anchor, embedding_positive)
    output = Dense(1, activation='sigmoid')(l1)

    return Model(inputs=[input_anchor, input_positive], outputs=output)

siamese_model = siamese_model()


siamese_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', Precision(), Recall()])


early_stopping = EarlyStopping(
    monitor='accuracy',
    mode='max',
    patience=3,
    restore_best_weights=True
)


history = siamese_model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    callbacks=[early_stopping]
)

print("Training completed successfully.")

# Save and reload the model
siamese_model.save(r'C:\Users\hp\OneDrive\Desktop\codes\project\Siamese_model.keras')
model = tf.keras.models.load_model(
    r'C:\Users\hp\OneDrive\Desktop\codes\project\Siamese_model.keras',
    custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy}
)
print("Model loaded successfully.")

test_anchor = tf.data.Dataset.list_files(ANC_PATH + '/*.jpg').skip(train_samples)
test_positive = tf.data.Dataset.list_files(POS_PATH + '/*.jpg').skip(train_samples)
test_negative = tf.data.Dataset.list_files(NEG_PATH + '/*.jpg').skip(train_samples)

test_positives = tf.data.Dataset.zip((test_anchor, test_positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(test_anchor)))))
test_negatives = tf.data.Dataset.zip((test_anchor, test_negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(test_anchor)))))
test_data = test_positives.concatenate(test_negatives)

test_data = test_data.map(preprocess_twin)
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

test_results = siamese_model.evaluate(test_data)
print(f"Test Results - Loss: {test_results[0]:.4f}, Accuracy: {test_results[1]:.4f}, Precision: {test_results[2]:.4f}, Recall: {test_results[3]:.4f}")

def verify(model,detection_threshold,verification_threshold):
    results=[]
    for image in os.listdir(os.path.join('application_data','verification_images')):
        input_img=preprocess(os.path.join('application_data','input_image','input_image.jpg'),True)
        validation_img=preprocess(os.path.join('application_data','verification_images',image),True)
        result=model.predict(list(np.expand_dims([input_img,validation_img],axis=1)))
        results.append(result)

    detection=np.sum(np.array(results)>detection_threshold)
    verification=detection/len(os.listdir(os.path.join('application_data','verification_images')))
    verified=verification>verification_threshold

    return results,verified
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break
    frame = frame[120:120+250, 200:200+250, :]
    cv2.imshow('Verification', frame)
    key = cv2.waitKey(10) & 0xFF
    #Verification trigger
    if key==ord('v'):
        cv2.imwrite(os.path.join('application_data','input_image','input_image.jpg'),frame)
        results,verified=verify(model,0.5,0.5 )
        print(verified)

    if key == ord('q'):
        print("Exiting data collection...")
        break

cap.release()
cv2.destroyAllWindows()