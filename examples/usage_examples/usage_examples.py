# %% Usage of AllInOneTuning
import parent_import
from tune_easy import AllInOneTuning
import seaborn as sns
# Load dataset
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # Select 2 classes
TARGET_VARIABLE = 'species'  # Target variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
y = iris[TARGET_VARIABLE].values
X = iris[USE_EXPLANATORY].values
###### Run All-in-one Tuning######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
all_tuner.df_scores

# %% Usage of each estimater's Tuning class
import parent_import
from tune_easy import LGBMClassifierTuning
import seaborn as sns
# Load dataset
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # Select 2 classes
TARGET_VARIABLE = 'species'  # Target variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
y = iris[TARGET_VARIABLE].values
X = iris[USE_EXPLANATORY].values
###### Run Detailed Tuning######
tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY)  # Initialize tuning instance
tuning.plot_first_validation_curve(cv=2)  # Plot first validation curve
tuning.optuna_tuning(cv=2)  # Optimization using Optuna library
tuning.plot_search_history()  # Plot score increase history
tuning.plot_search_map()  # Visualize relationship between parameters and validation score
tuning.plot_best_learning_curve()  # Plot learning curve
tuning.plot_best_validation_curve()  # Plot validation curve

# %% MLflow example
import parent_import
from tune_easy import AllInOneTuning
import seaborn as sns
# Load dataset
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # Select 2 classes
TARGET_VARIABLE = 'species'  # Target variable
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
y = iris[TARGET_VARIABLE].values
X = iris[USE_EXPLANATORY].values
###### Run All-in-one Tuning with MLflow logging ######
all_tuner = AllInOneTuning()
all_tuner.all_in_one_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2,
                           mlflow_logging=True)  # Set MLflow logging argument
# %%
