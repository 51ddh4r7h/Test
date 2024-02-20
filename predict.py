import tensorflow as tf
import matplotlib.pyplot as plt
from src.model import create_model
from src.utils import load_and_preprocess_fashion_mnist

# Load your model (assuming you have 'my_model.h5')
model = tf.keras.models.load_model('trained_models/my_model.h5')  

# Preprocess some sample data  
(train_images, train_labels), (test_images, test_labels) = load_and_preprocess_fashion_mnist()
sample_input = test_images[0:1]

# Use the model to make a prediction
prediction = model.predict(sample_input)

# Get the predicted class label
predicted_class = tf.argmax(prediction, axis=1).numpy()[0]

# Define class names for Fashion MNIST dataset
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# Plot the input image along with its predicted class label
plt.figure(figsize=(6, 3))
plt.imshow(sample_input[0], cmap=plt.cm.binary)
plt.title(f'Predicted Class: {class_names[predicted_class]}')
plt.axis('off')
plt.show()
