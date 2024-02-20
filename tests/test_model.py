import unittest
import tensorflow as tf
from src.model import create_model 
from src.utils import load_and_preprocess_fashion_mnist

class TestModel(unittest.TestCase):
    def test_model_output_shape(self):
        model = create_model()
        # Load some sample data 
        (_, _), (test_images, _) = load_and_preprocess_fashion_mnist()
        sample_input = test_images[0:1]  # Take the first image

        output = model(sample_input)
        self.assertEqual(output.shape, (1, 10))  

if __name__ == '__main__':
    unittest.main()
