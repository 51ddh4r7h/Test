import tensorflow as tf

from src.utils import load_and_preprocess_fashion_mnist

def create_model(): 
    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_fashion_mnist() 

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model 
