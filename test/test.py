import sys
sys.path.append('/path/to/ml_model')
from ml_model.train import train_model
from ml_model.train import train_model
def test_train_model():
 # Train the model
 model = train_model()
 assert model is not None # Ensure the model is trained
def test_prediction():
 # Test if prediction works with the model
 from ml_model.predict import predict
 sample_input = [0.00632, 18.0, 2.31, 0.0, 0.538, 6.575, 65.2, 4.0900, 1.0, 296.0]
 prediction = predict(sample_input)
 assert len(prediction) == 1 # Ensure prediction is returned 
