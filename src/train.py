import os  # We'll need the 'os' module 
import tensorflow as tf
from src.model import create_model
from src.utils import load_and_preprocess_fashion_mnist 

def main():
    # Load and preprocess data
    (train_images, train_labels), (test_images, test_labels) = load_and_preprocess_fashion_mnist() 

    model = create_model()
    model.fit(train_images, train_labels, epochs=5)

    test_loss, test_acc = model.evaluate(test_images,  test_labels)
    print('Test accuracy:', test_acc)

    # Create the 'trained_models' folder if it doesn't exist
    trained_models_dir = 'trained_models'
    os.makedirs(trained_models_dir, exist_ok=True) 

    # Save the model 
    model_save_path = os.path.join(trained_models_dir, 'my_model.h5')
    model.save(model_save_path)  

if __name__ == "__main__":
    main()
