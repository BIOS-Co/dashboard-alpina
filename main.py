from explainerdashboard import ClassifierExplainer, ExplainerDashboard, ExplainerHub
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import joblib

# Load the data

X = pd.read_csv("Data/DATOS_CLINICOS (1).CSV", index_col = 0, sep=";")
X.columns = [X.columns[i].replace(".","_") for i in range(X.shape[1])]
print(X)
# Load the pipelines

model_metabolismo = joblib.load("Models/clasificator_metabolismo.joblib")


# Define the explainers

explainer_metabolismo = ClassifierExplainer(model_metabolismo, X, labels = ["Sin Dislipidemia", "Dislipidemia"])

# Define the dashboards

db_metabolismo = ExplainerDashboard(explainer_metabolismo, title="Riesgo Metabólico", description="")

# Define the hub:

hub = ExplainerHub([db_metabolismo], title = "Analítica predictiva", description= "Tableros de analítica predictiva para predicción de riesgo")
hub.run(port=2021)
