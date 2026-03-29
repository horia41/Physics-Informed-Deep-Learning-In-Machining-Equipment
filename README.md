# Physics Informed Deep Learning In Machining Equipment (PINNs)

Some updates: 
- folder `dataset` handles everything in terms of data ; has `download_data.py` which automatically downloads the MATWI dataset and creates a folder `dataset/matwi` where it will be downloaded; has `EDA.py` which will output in a folder `dataset/eda_output` and performs exploratory data analysis on our dataset which will help us see details about our data and establish what data preprocessing steps and feature engineering we should perform; here, still missing the DataLoader part
- created folders `vision-baseline`, `vision-sensor` and `pinn` where we'll develop each model accordingly so we dont necessarily have to create multiple branches while using the same `dataset` folder

So, for now, make sure to run the `dataset/download_data.py` and then the `dataset/EDA.py` so you can see the plots and outputs of it.