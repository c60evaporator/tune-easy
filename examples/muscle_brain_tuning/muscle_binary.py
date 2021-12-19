# %% MuscleTuning, binary, no argument
import parent_import
from muscle_tuning import MuscleTuning
import seaborn as sns
# Load dataset
iris = sns.load_dataset("iris")
iris = iris[iris['species'] != 'setosa']  # Only 2 class
TARGET_VARIALBLE = 'species'  # Target variable name
USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Selected explanatory variables
y = iris[TARGET_VARIALBLE].values
X = iris[USE_EXPLANATORY].values
# Run parameter tuning
kinnikun = MuscleTuning()
kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY)
kinnikun.df_scores

# %% MuscleTuning, binary, all arguments
