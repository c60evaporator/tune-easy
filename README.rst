=========
tune-easy
=========

|python| |pypi| |license|

.. |python| image:: https://img.shields.io/pypi/pyversions/tune-easy
   :target: https://www.python.org/

.. |pypi| image:: https://img.shields.io/pypi/v/tune-easy?color=blue
   :target: https://pypi.org/project/tune-easy/

.. |license| image:: https://img.shields.io/pypi/l/tune-easy?color=blue
   :target: https://github.com/c60evaporator/tune-easy/blob/master/LICENSE

A hyperparameter tuning tool, extremely easy to use.

.. image:: https://user-images.githubusercontent.com/59557625/165905780-d153541a-6c74-4dc6-a37f-7d63151bf582.png
   :width: 320px

.. image:: https://user-images.githubusercontent.com/59557625/145702196-50f6781e-2ca2-4cbf-9344-ab58cb08d34b.png
   :width: 540px

This package supports scikit-learn API estimators, such as SVM and LightGBM.

=====
Usage
=====

Example of All-in-one Tuning
============================
If you want to optimize multiple estimators simultaneously, Use "All-in-one Tuning"

.. code-block:: python

    from tune_easy import AllInOneTuning
    import seaborn as sns
    # Load Dataset
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

.. image:: https://user-images.githubusercontent.com/59557625/165905780-d153541a-6c74-4dc6-a37f-7d63151bf582.png
   :width: 320px

.. image:: https://user-images.githubusercontent.com/59557625/145702196-50f6781e-2ca2-4cbf-9344-ab58cb08d34b.png
   :width: 480px

If you want to know the usage in detail, see `API Reference
<https://c60evaporator.github.io/tune-easy/all_in_one_tuning.html>`__ and `Examples
<https://github.com/c60evaporator/tune-easy/tree/master/examples/all_in_one_tuning>`__

Example of Detailed Tuning
==========================
If you want to optimize one estimator in detail, Use "Detailed Tuning"

.. code-block:: python

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

.. image:: https://user-images.githubusercontent.com/59557625/145702586-8b341344-625c-46b3-a9ee-89cb592b1800.png
   :width: 320px

.. image:: https://user-images.githubusercontent.com/59557625/145702594-cc4b2194-2ed0-40b0-8a83-94ebd8162818.png
   :width: 480px

.. image:: https://user-images.githubusercontent.com/59557625/145702643-70e3b1f2-66aa-4619-9703-57402b3669aa.png
   :width: 320px

If you want to know the usage in detail, see `API Reference
<https://c60evaporator.github.io/tune-easy/each_estimators.html>`__ and `Examples
<https://github.com/c60evaporator/tune-easy/blob/master/examples/usage_examples/userguide_examples_each.ipynb>`__

Example of MLflow logging
=========================

.. code-block:: python

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

.. code-block:: console

    $ mlflow ui

.. image:: https://user-images.githubusercontent.com/59557625/147270240-f779cf1f-b216-42a2-8156-37169511ec3e.png
   :width: 800px

If you want to know the usage in detail, see `API Reference
<https://c60evaporator.github.io/tune-easy/all_in_one_tuning.html#tune_easy.all_in_one_tuning.AllInOneTuning.all_in_one_tuning>`__ and `Examples
<https://github.com/c60evaporator/tune-easy/tree/master/examples/mlflow>`__


============
Requirements
============
param-tuning-utility 0.2.1 requires

* Python >=3.6
* Scikit-learn >=0.24.2
* Numpy >=1.20.3
* Pandas >=1.2.4
* Matplotlib >=3.3.4
* Seaborn >=0.11.0
* Optuna >=2.7.0
* BayesianOptimization >=1.2.0
* MLFlow >=1.17.0
* LightGBM >=3.3.2
* XGBoost >=1.4.2
* seaborn-analyzer >=0.2.11

====================
Installing tune-easy
====================
Use pip to install the binary wheels on `PyPI <https://pypi.org/project/tune-easy/>`__

.. code-block:: console

    $ pip install tune-easy

=======
Support
=======
Bugs may be reported at https://github.com/c60evaporator/tune-easy/issues

=============
API Reference
=============
The following classes are included in tune-easy

All-in-one Tuning
=================

.. csv-table::
    :header: "Class name", "Summary", "API Documentation", "Example"
    :widths: 30, 50, 15, 15

    "**AllInOneTuning**", To optimize multiple estimators simultaneously., `tune_easy.all_in_one_tuning.AllInOneTuning <https://c60evaporator.github.io/tune-easy/all_in_one_tuning.html#tune_easy.all_in_one_tuning.AllInOneTuning.all_in_one_tuning>`__, `example <https://github.com/c60evaporator/tune-easy/tree/master/examples/all_in_one_tuning>`__


Detailed Tuning
===============
Select the following class corresponds to using estimator.

.. csv-table::
    :header: "Class name", "Summary", "API Documentation", "Example"
    :widths: 30, 30, 25, 25

    "**LGBMRegressorTuning**", For LGBMRegressor, `tune_easy.lgbm_tuning.LGBMRegressorTuning <https://c60evaporator.github.io/tune-easy/each_estimators.html#tune_easy.lgbm_tuning.LGBMRegressorTuning>`__, `example <https://github.com/c60evaporator/tune-easy/blob/master/examples/regression_california/example_lgbm_california.py>`__
    "**XGBRegressorTuning**", For XGBRegressor, `tune_easy.xgb_tuning.XGBRegressorTuning <https://c60evaporator.github.io/tune-easy/each_estimators.html#tune_easy.xgb_tuning.XGBRegressorTuning>`__, `example <https://github.com/c60evaporator/tune-easy/blob/master/examples/regression_california/example_xgb_california.py>`__
    "**SVMRegressorTuning**", For svr, `tune_easy.svm_tuning.SVMRegressorTuning <https://c60evaporator.github.io/tune-easy/each_estimators.html#tune_easy.svm_tuning.SVMRegressorTuning>`__, `example <https://github.com/c60evaporator/tune-easy/blob/master/examples/regression_california/example_svr_california.py>`__
    "**RFRegressorTuning**", For RandomForestRegressor, `tune_easy.rf_tuning.RFRegressorTuning <https://c60evaporator.github.io/tune-easy/each_estimators.html#tune_easy.rf_tuning.RFRegressorTuning>`__, `example <https://github.com/c60evaporator/tune-easy/blob/master/examples/regression_california/example_rf_california.py>`__
    "**ElasticNetTuning**", For ElasticNet, `tune_easy.elasticnet_tuning.ElasticNetTuning <https://c60evaporator.github.io/tune-easy/each_estimators.html#tune_easy.elasticnet_tuning.ElasticNetTuning>`__, `example <https://github.com/c60evaporator/tune-easy/blob/master/examples/regression_california/example_elasticnet_california.py>`__
    "**LGBMClassifierTuning**", For LGBMClassifier, `tune_easy.lgbm_tuning.LGBMClassifierTuning <https://c60evaporator.github.io/tune-easy/each_estimators.html#tune_easy.lgbm_tuning.LGBMClassifierTuning>`__, `example <https://github.com/c60evaporator/tune-easy/blob/master/examples/classification_iris/example_lgbm_iris.py>`__
    "**XGBClassifierTuning**", For XGBClassifier, `tune_easy.xgb_tuning.XGBClassifierTuning <https://c60evaporator.github.io/tune-easy/each_estimators.html#tune_easy.xgb_tuning.XGBClassifierTuning>`__, `example <https://github.com/c60evaporator/tune-easy/blob/master/examples/classification_iris/example_xgb_iris.py>`__
    "**SVMClassifierTuning**", For svm, `tune_easy.svm_tuning.SVMClassifierTuning <https://c60evaporator.github.io/tune-easy/each_estimators.html#tune_easy.svm_tuning.SVMClassifierTuning>`__, `example <https://github.com/c60evaporator/tune-easy/blob/master/examples/classification_iris/example_svm_iris.py>`__
    "**RFClassifierTuning**", For RandomForestClassifier, `tune_easy.rf_tuning.RFClassifierTuning <https://c60evaporator.github.io/tune-easy/each_estimators.html#tune_easy.rf_tuning.RFClassifierTuning>`__, `example <https://github.com/c60evaporator/tune-easy/blob/master/examples/classification_iris/example_rf_iris.py>`__
    "**LogisticRegressionTuning**", For LogisticRegression, `tune_easy.logisticregression_tuning.LogisticRegressionTuning <https://c60evaporator.github.io/tune-easy/each_estimators.html#tune_easy.logisticregression_tuning.LogisticRegressionTuning>`__, `example <https://github.com/c60evaporator/tune-easy/blob/master/examples/classification_iris/example_logistic_iris.py>`__
