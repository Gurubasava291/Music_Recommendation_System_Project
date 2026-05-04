from sklearn.ensemble import RandomForestClassifier

class RFClassifier:
    def __init__(self, n_estimators=100, max_depth=None, random_state=42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, 
            max_depth=max_depth, 
            random_state=random_state
        )

    def fit(self, X, y):
        """
        Train the Random Forest model.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Predict classes using the Random Forest model.
        """
        return self.model.predict(X)

    def score(self, X, y):
        """
        Evaluate the model and return the accuracy score.
        """
        return self.model.score(X, y)
