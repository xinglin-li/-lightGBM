# LightGBM Custom Node for SAS Visual Forecasting

This repository contains code and zip files for multiple versions of a custom Light Gradient Boosting Machine (LGBM) node for SAS Visual Forecasting. The zip files include `code.sas`, `validation.xml`, `template.json`, and `metadata.json`. You can upload the zip files to "The Exchange" in SAS Visual Forecasting to use the custom nodes. For details on creating a Gradient Boosting-based Forecasting node, refer to this [paper](https://support.sas.com/resources/papers/proceedings19/3258-2019.pdf). 

Stay tuned for forthcoming blogs that will illustrate the functionality of these custom LGBM nodes.

## Basic Code
The basic code forms the backbone of the custom LGBM node without any additional extensions. Details will be available in the upcoming blog.

- **GBM_TSMODEL.sas**: Modifies the previous custom Gradient Boosting Machine node. The forecasting process is based on `PROC ASTORE` and `PROC TSMODEL`.
- **LGBM_TSMODEL.sas**: Contains the code for the custom LightGBM node. The forecasting process is based on `PROC ASTORE` and `PROC TSMODEL`.
- **LGBM_SQL.sas**: Implements the custom LightGBM node using SQL. Instead of using `PROC TSMODEL` for filling values in the forecasting process, it uses SQL to merge tables and fill missing values for future Y lags. The efficiency of this approach is similar to the previous one. 

## LGBM Node with Additional Options
These custom nodes extend the basic functionality by introducing additional options.

- **lgbm_with_fit_forecast_options.sas**: Provides options for both fitting and forecasting. The model can be trained under "fit" mode, and you can reuse the trained model for forecasting by setting the task to "forecast."
- **lgbm_with_fit_forecast_reuse_options.sas**: Adds an option to reuse predictions from the training task in subsequent forecasts. However, this requires storing previous outputs, resulting in higher memory consumption compared to the version without this option.

## LGBM Node with Feature Extraction
These custom nodes integrate open-source package "tsfresh" for time series feature extraction. There are two methods for extending the extracted features into the forecasting horizon:
1. **TSDF Method**: Extends the features in one step using the TSDF package. This method is faster.
2. **Iterative Method**: Extends the features iteratively by updating the features each time a new prediction is made. This method is theoretically more accurate but takes longer to run.

- **LGBM_FE_TSDF_EXT.sas**: Extends extracted features using the TSDF package in `PROC TSMODEL`.
- **LGBM_FE_ITER_EXT.sas**: Extends features iteratively over the forecasting horizon. Each prediction is followed by feature extraction using the predicted values, iteratively updating the dataset.

More details on these methods will be discussed in the upcoming blog.

