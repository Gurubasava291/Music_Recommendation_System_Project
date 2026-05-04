import mlflow
import mlflow.pytorch
import pandas as pd
from src.dataset_prep import load_and_split_data, get_mdp_states
from src.dqn_agent import DQNAgent
import numpy as np

def train_model():
    print("Starting MLOps Training Pipeline...")
    
    # 1. Load Data
    train_df, test_df = load_and_split_data('data/dataset.csv', test_ratio=0.2)
    STATE_LENGTH = 3
    NUM_SONGS = 200
    train_mdp = get_mdp_states(train_df, state_length=STATE_LENGTH)
    
    # 2. Setup MLflow tracking
    mlflow.set_experiment("Music_Recommender_DQN")
    
    with mlflow.start_run():
        # Log parameters
        epochs = 10
        alpha = 0.001
        gamma = 0.9
        epsilon = 0.2
        mlflow.log_params({"epochs": epochs, "lr": alpha, "gamma": gamma, "epsilon": epsilon})
        
        # 3. Initialize Agent
        agent = DQNAgent(num_songs=NUM_SONGS, state_length=STATE_LENGTH, lr=alpha, gamma=gamma, epsilon=epsilon)
        
        # 4. Train Model
        print("Training Deep Q-Network...")
        for epoch in range(epochs):
            epoch_loss = 0
            epoch_reward = 0
            np.random.shuffle(train_mdp)
            
            # Simple batching approximation
            for user_id, state, action, reward in train_mdp[:1000]: # Train on a subset for speed in this demo
                next_state = list(state[1:]) + [action]
                loss = agent.update(state, action, reward, next_state)
                epoch_loss += loss
                epoch_reward += reward
                
            # Log metrics
            avg_loss = epoch_loss / 1000
            mlflow.log_metric("loss", avg_loss, step=epoch)
            mlflow.log_metric("reward", epoch_reward, step=epoch)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Reward: {epoch_reward}")
            
        # 5. Save Model
        mlflow.pytorch.log_model(agent.model, "dqn_model")
        print("DQN Training complete. Model logged to MLflow.")

    # Train DNN Classifier
    mlflow.set_experiment("Music_Recommender_DNN")
    with mlflow.start_run():
        from src.dnn_classifier import DNNClassifier
        import torch
        
        # Mock dataset for DNN (e.g. predicting user segment from 2 features)
        X_train = torch.rand(1000, 2)
        y_train = torch.randint(0, 2, (1000,)).float()
        X_test = torch.rand(200, 2)
        y_test = torch.randint(0, 2, (200,)).float()
        
        dnn_model = DNNClassifier(input_dim=2, hidden_dim=16, layers=2)
        print("Training DNN Classifier...")
        
        # Use the newly added .fit() method
        dnn_model.fit(X_train, y_train, epochs=10, batch_size=32)
        
        # Use the newly added .score() method
        accuracy = dnn_model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.pytorch.log_model(dnn_model.model, "dnn_model")
        print(f"DNN Training complete. Test Accuracy: {accuracy:.4f}")

    # Train Random Forest Classifier
    mlflow.set_experiment("Music_Recommender_RF")
    with mlflow.start_run():
        from src.rf_classifier import RFClassifier
        import numpy as np
        
        # Mock dataset for RF (same structure as DNN)
        X_train_rf = np.random.rand(1000, 2)
        y_train_rf = np.random.randint(0, 2, 1000)
        X_test_rf = np.random.rand(200, 2)
        y_test_rf = np.random.randint(0, 2, 200)
        
        rf_model = RFClassifier(n_estimators=100)
        print("Training Random Forest Classifier...")
        
        rf_model.fit(X_train_rf, y_train_rf)
        
        accuracy_rf = rf_model.score(X_test_rf, y_test_rf)
        mlflow.log_metric("accuracy", accuracy_rf)
        mlflow.sklearn.log_model(rf_model.model, "rf_model")
        print(f"RF Training complete. Test Accuracy: {accuracy_rf:.4f}")

if __name__ == "__main__":
    train_model()
