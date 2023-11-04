# customer-shopping-price

'''
project_name
│
├── __init__.py
│
├── components
│   └── __init__.py
│
├── config
│   └── __init__.py
│
├── constants
│   └── __init__.py
│
├── entity
│   └── __init__.py
│
├── exception
│   └── __init__.py
│
├── logger
│   └── __init__.py
│
├── pipeline
│   └── __init__.py
│
├── Utils
│   └── __init__.py
│
├── schema.yaml
├── app.py
├── logs.py
├── exception.py
├── setup.py
├── requirements.txt
└── pipeline.txt

'''



ptoject flow:
1. creating virtual environment
2. setup file creation
3. defining constants in constant folder
4. configuring constant in configuration file
5. data_injection 
6. data_transformation
7. model_training

## web aplication
# Sculpture Shopping Price Prediction App

This Streamlit app allows users to predict sculpture prices based on various features. Whether you want to analyze a single sculpture or make bulk predictions, this app has got you covered!

## Key Features:

- **Train a Model:** Explore the machine learning model training process for predicting sculpture prices. This is a demonstration page and does not require uploading a training dataset.

- **Batch Prediction:** Predict prices for multiple sculptures at once by uploading a CSV file.

- **Single Prediction:** Predict the price of a single sculpture by providing its details.

## How to Use:

1. **Train a Model:**
   - Click on 'Train Model' in the sidebar.
   - Upload your training dataset and start training.

2. **Batch Prediction:**
   - Select 'Batch Prediction' in the sidebar.
   - Upload a CSV file with sculpture details.
   - View the predictions in a table.

3. **Single Prediction:**
   - Select 'Single Prediction' in the sidebar.
   - Enter the sculpture details.
   - Click 'Predict' to get the estimated price.

## Sample Dataset:

To get started, you can use the sample dataset provided in the [`data`](data) directory. The sample dataset contains example input data in CSV format. You can download the sample dataset from the following link:
[Download Sample Batch Prediction Dataset](https://github.com/kalehariprasad/sculpture-shopping-price/tree/main/data)

The sample dataset provides a template for the input file format expected by the app. You can open the CSV file to see the example input data and use it as a reference when preparing your own dataset for batch prediction.

## Need Help?

If you encounter any issues or have questions, feel free to reach out on the [GitHub repository](https://github.com/kalehariprasad/sculpture-shopping-price). I am here to assist you!

## Ready to Get Started?

**👈 Select a demo from the sidebar** to begin exploring the app's functionalities!
