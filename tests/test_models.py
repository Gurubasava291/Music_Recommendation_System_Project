import pytest
import torch
from src.dqn_agent import DQN, DQNAgent
from src.dnn_classifier import DNNClassifier
from src.qml_classifier import QMLClassifier

def test_dqn_forward():
    num_songs = 200
    state_length = 3
    model = DQN(num_songs, state_length)
    
    # Batch size 2
    dummy_input = torch.tensor([[10, 20, 30], [5, 15, 25]], dtype=torch.long)
    output = model(dummy_input)
    
    assert output.shape == (2, num_songs)
    
def test_dnn_classifier_predict():
    classifier = DNNClassifier(layers=1)
    
    # 2 features per sample
    dummy_input = torch.tensor([[0.5, 0.2], [0.1, 0.9]], dtype=torch.float32)
    output = classifier.predict(dummy_input)
    
    assert output.shape == (2,)
    # Output should be probability between 0 and 1
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)

def test_qml_classifier_predict():
    classifier = QMLClassifier(layers=1)
    
    # 2 features per sample
    dummy_input = torch.tensor([[0.5, 0.2], [0.1, 0.9]], dtype=torch.float32)
    output = classifier.predict(dummy_input)
    
    assert output.shape == (2,)
    # Output should be probability between 0 and 1
    assert torch.all(output >= 0.0)
    assert torch.all(output <= 1.0)
