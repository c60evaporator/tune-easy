====================
param-tuning-utility
====================
A hyperparameter tuning tool with gorgeous UI for scikit-learn API

=====
Usage
=====

Example of All-in-one Tuning
============================

.. code-block:: python

    from muscle_tuning import MuscleTuning
    import seaborn as sns
    # Load Dataset
    iris = sns.load_dataset("iris")
    iris = iris[iris['species'] != 'setosa']  # Select 2 classes
    TARGET_VARIALBLE = 'species'  # Target variable
    USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
    y = iris[OBJECTIVE_VARIALBLE].values
    X = iris[USE_EXPLANATORY].values
    ###### Run All-in-one Tuning######
    kinnikun = MuscleTuning()
    kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2)
    kinnikun.df_scores

.. image:: https://user-images.githubusercontent.com/59557625/140383755-bca64ab3-1593-47ef-8401-affcd0b20a0a.png
   :width: 320px

.. image:: https://user-images.githubusercontent.com/59557625/145702196-50f6781e-2ca2-4cbf-9344-ab58cb08d34b.png
   :width: 480px

If you want to know usage of the other classes, see `API Reference
<https://c60evaporator.github.io/muscle-tuning/muscle_tuning.html>`__ and `Examples
<https://github.com/c60evaporator/muscle-tuning/tree/master/examples/muscle_brain_tuning>`__

Example of Detailed Tuning
==========================

.. code-block:: python

    from muscle_tuning import LGBMClassifierTuning
    from sklearn.datasets import load_boston
    import seaborn as sns
    # データセット読込
    iris = sns.load_dataset("iris")
    iris = iris[iris['species'] != 'setosa']  # Select 2 classes
    OBJECTIVE_VARIALBLE = 'species'  # Target variable
    USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
    y = iris[OBJECTIVE_VARIALBLE].values
    X = iris[USE_EXPLANATORY].values
    ###### Run Detailed Tuning######
    tuning = LGBMClassifierTuning(X, y, USE_EXPLANATORY)  # Initialize tuning instance
    tuning.plot_first_validation_curve(cv=2)  # Plot first validation curve
    tuning.optuna_tuning(cv=2)  # Optimization using Optuna library
    tuning.plot_search_history()  # Plot score increase history
    tuning.plot_search_map()  # Visualize relationship between parameters and validation score
    tuning.plot_best_learning_curve()  # Plot learning curve
    tuning.plot_best_validation_curve()  # Plot validation curve

.. image:: https://user-images.githubusercontent.com/59557625/145702586-8b341344-625c-46b3-a9ee-89cb592b1800.png
   :width: 320px

.. image:: https://user-images.githubusercontent.com/59557625/145702594-cc4b2194-2ed0-40b0-8a83-94ebd8162818.png
   :width: 480px

.. image:: https://user-images.githubusercontent.com/59557625/145702643-70e3b1f2-66aa-4619-9703-57402b3669aa.png
   :width: 320px

If you want to know usage of the other classes, see `API Reference
<https://c60evaporator.github.io/muscle-tuning/each_estimators.html>`__ and `Examples
<https://github.com/c60evaporator/muscle-tuning/tree/master/examples/method_examples>`__

Example of MLflow logging
=========================

.. code-block:: python

    from muscle_tuning import MuscleTuning
    import seaborn as sns
    # Load dataset
    iris = sns.load_dataset("iris")
    iris = iris[iris['species'] != 'setosa']  # Select 2 classes
    TARGET_VARIALBLE = 'species'  # Target variable
    USE_EXPLANATORY = ['petal_width', 'petal_length', 'sepal_width', 'sepal_length']  # Explanatory variables
    y = iris[TARGET_VARIALBLE].values
    X = iris[USE_EXPLANATORY].values
    ###### Run All-in-one Tuning with MLflow logging ######
    kinnikun = MuscleTuning()
    kinnikun.muscle_brain_tuning(X, y, x_colnames=USE_EXPLANATORY, cv=2,
                                 mlflow_logging=True)  # Set MLflow logging argument

.. image:: https://user-images.githubusercontent.com/59557625/145711588-be0e393f-be7b-4833-b17a-05eecd6ad014.png
   :width: 640px

If you want to know usage of the other classes, see `API Reference
<https://c60evaporator.github.io/muscle-tuning/muscle_tuning.html#muscle_tuning.muscle_tuning.MuscleTuning.muscle_brain_tuning>`__ and `Examples
<https://github.com/c60evaporator/muscle-tuning/tree/master/examples/mlflow>`__


============
Requirements
============
param-tuning-utility 0.1.10 requires

* Python >=3.6
* Scikit-learn >=0.24.2
* Numpy >=1.20.3
* Pandas >=1.2.4
* Matplotlib >=3.3.4
* Seaborn >=0.11.0
* Optuna >=2.7.0
* BayesianOptimization >=1.2.0
* MLFlow >=1.17.0
* LightGBM >=3.2.1
* XGBoost >=1.4.2

========================
Installing muscle-tuning
========================
Use pip to install the binary wheels on `PyPI <https://pypi.org/project/muscle-tuning/>`__

.. code-block:: console

    $ pip install muscle-tuning

=======
Support
=======
Bugs may be reported at https://github.com/c60evaporator/muscle-tuning/issues
