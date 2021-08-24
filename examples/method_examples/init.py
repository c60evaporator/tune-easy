# %% __init__(), no argument
import parent_import
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# Load dataset
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']  # Feature names used as explanatory variables
df_boston = pd.DataFrame(load_boston().data, columns = load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values  # Explanatory variables
y = load_boston().target  # Objective variable
###### __init() ######
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)

# %% __init__(), for LeaveOneGroupOut
import parent_import
from param_tuning import XGBRegressorTuning
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
# データセット読込
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'
USE_EXPLATATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
X = df_reg[USE_EXPLATATORY].values  # Explanatory variables
y = df_reg[OBJECTIVE_VARIABLE].values  # Objective variable
###### __init() ######
tuning = XGBRegressorTuning(X, y, USE_EXPLANATORY,  # Required argument
                            cv_group=df_reg['ward_after'].values)  # Grouping data for LeaveOneGroupOut

# %% __init__(), use validation data as eval_data in fit_params
import parent_import
from param_tuning import LGBMRegressorTuning
from sklearn.datasets import load_boston
import pandas as pd
# Load dataset
USE_EXPLANATORY = ['CRIM', 'NOX', 'RM', 'DIS', 'LSTAT']
df_boston = pd.DataFrame(load_boston().data, columns = load_boston().feature_names)
X = df_boston[USE_EXPLANATORY].values  # Explanatory variables
y = load_boston().target  # Objective variable
###### __init() ######
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY,  # Required argument
                             eval_data_source='valid')  # Use valid data as eval_set