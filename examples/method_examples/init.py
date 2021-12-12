# %% __init__(), no argument
import parent_import
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # Objective variable
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
y = df_reg[OBJECTIVE_VARIABLE].values
X = df_reg[USE_EXPLANATORY].values
###### __init() ######
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY)

# %% __init__(), for LeaveOneGroupOut
import parent_import
from muscle_tuning import XGBRegressorTuning
from sklearn.model_selection import LeaveOneGroupOut
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # Objective variable
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
X = df_reg[USE_EXPLANATORY].values
y = df_reg[OBJECTIVE_VARIABLE].values
###### __init() ######
tuning = XGBRegressorTuning(X, y, USE_EXPLANATORY,  # Required argument
                            cv_group=df_reg['ward_after'].values)  # Grouping data for LeaveOneGroupOut

# %% __init__(), use validation data as eval_data in fit_params
import parent_import
from muscle_tuning import LGBMRegressorTuning
import pandas as pd
# Load dataset
df_reg = pd.read_csv(f'../sample_data/osaka_metropolis_english.csv')
OBJECTIVE_VARIABLE = 'approval_rate'  # Objective variable
USE_EXPLANATORY = ['2_between_30to60', '3_male_ratio', '5_household_member', 'latitude']  # Explanatory variables
X = df_reg[USE_EXPLANATORY].values
y = df_reg[OBJECTIVE_VARIABLE].values
###### __init() ######
tuning = LGBMRegressorTuning(X, y, USE_EXPLANATORY,  # Required argument
                             eval_data_source='valid')  # Use valid data as eval_set
# %%
