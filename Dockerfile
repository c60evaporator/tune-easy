FROM python:3.8-buster

# 必要ライブラリのインストール
RUN pip install \
         pandas==1.2.0 \
         numpy==1.19.3 \
         seaborn==0.11.1 \
         xgboost==1.3.1 \
         scikit-learn==0.24.0 \
         bayesian-optimization==1.2.0 \
         jupyter==1.0.0 \
         mlxtend==0.18.0

# スクリプトをclone
RUN mkdir -p /opt/Programs/Python
WORKDIR /opt/Programs/Python
RUN git clone https://github.com/c60evaporator/param_tuning_utility.git