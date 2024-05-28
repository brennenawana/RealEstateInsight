from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import joblib

data = load_iris()
X, y = data.data, data.target
model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'model.pkl')
