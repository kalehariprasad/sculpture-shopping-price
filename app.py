
from src.constants import *
from src.config.configuration import *
import os,sys
from typing import Any
import pandas as pd
import numpy as np
import streamlit as st
from src.logger import logging
from src.exception import CustomException
from src.components.data_injection import DataInjection
from src.components.data_transformation import DataTransformation
from src.components.model_training import Model_Trainer
from prediction.batch import BatchPrediction
from src.pipeline.prediction_pipeline import CustomData,PredictionPipeline
from src.pipeline.training_pipeline import Train


from src.Utils import load_model

class App:
    def __init__(self):
     self.pages = {
    "Home": self.home,
    "Train": self.train,
    "Batch Prediction": self.batch_prediction,
    "Single Prediction": self.single_prediction
}

    def home(self):
        st.header("Predict Sculpture Prices with Ease")
        st.write("This Streamlit app empowers you to predict sculpture prices based on various features. "
                "Whether you want to analyze a single sculpture or make bulk predictions, we've got you covered!")
        
        st.write("### Key Features:")
        st.markdown("- **Train a Model:** Explore the machine learning model training process for predicting sculpture prices. This is a demonstration page and does not require uploading a training dataset."
                    "\n- **Batch Prediction:** Predict prices for multiple sculptures at once by uploading a CSV file."
                    "\n- **Single Prediction:** Predict the price of a single sculpture by providing its details.")
        
        st.write("### How to Use:")
        st.write("1. **Train a Model:** Click on 'Train Model' in the sidebar, upload your training dataset, and start training."
                "\n2. **Batch Prediction:** Select 'Batch Prediction' in the sidebar, upload a CSV file with sculpture details, "
                "and view the predictions in a table."
                "\n3. **Single Prediction:** Select 'Single Prediction' in the sidebar, enter the sculpture details, and click 'Predict' "
                "to get the estimated price.")
        
        # Provide detailed download instructions for the sample dataset hosted online
        sample_dataset_url = "https://github.com/kalehariprasad/sculpture-shopping-price/tree/main/data"
        st.write("To get started, you can use  sample dataset to test the batch prediction functionality. The sample dataset contains example input data in CSV format. You can download the sample dataset from the following link:")
        st.markdown(f"[Download Sample Batch Prediction Dataset]({sample_dataset_url})")
        st.write("The sample dataset provides a template for the input file format expected by the app. You can open the CSV file to see the example input data and use it as a reference when preparing your own dataset for batch prediction.")

        
        st.write("### Need Help?")
        st.write("If you encounter any issues or have questions, feel free to reach out on  [GitHub repository](https://github.com/kalehariprasad/sculpture-shopping-price). "
                "i am ' here to assist you!")

        st.write("### Ready to Get Started?")
        st.write("**👈 Select a demo from the sidebar** to begin exploring the app's functionalities!")
    def train(self):
        st.title('Sculpture Price Prediction Training Stage')
        training=Train()
        training.main()
        st.header('training compleeted')
      


    def batch_prediction(self):
        st.title("Batch Sculpture Price Prediction")
        st.write("Upload a CSV file containing multiple sculpture records to predict their prices in bulk. "
                "The predictions will be displayed as a table below.")
        batch_predictor = BatchPrediction() 
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            prediction = batch_predictor.main(df)
            st.write('predictions for the given dataset including the dataframe with predicted column')
            st.write(pd.DataFrame(prediction)) 
        
    def single_prediction(self):
        Artist_Reputation = st.number_input(label='Enter Artist Reputation 0.1 - 1', min_value=0.1, max_value=1.0)
        Height = st.number_input(label='Enter height of the sculpture')
        Material = st.selectbox(label='Select type of material used for sculpture', options=['Brass', 'Clay', 'Aluminium', 'Wood', 'Marble', 'Bronze', 'Stone'])
        Price_Of_Sculpture = st.number_input(label='Enter price of the sculpture in $', min_value=10)
        Base_Shipping_Price = st.number_input(label='Base shipping price of the sculpture in $', min_value=10)
        International = st.selectbox(label='Is shipping international?', options=['Yes', 'No'])
        Express_Shipment = st.selectbox(label='Is shipping Express?', options=['Yes', 'No'])
        Installation_Included = st.selectbox(label='Is installation included in shipping?', options=['Yes', 'No'])
        Transport = st.selectbox(label='Select type of Transport for the shipment', options=['Airways', 'Roadways', 'Waterways'])
        Fragile = st.selectbox('Is the shipment Fragile?', options=['Yes', 'No'])
        Customer_Information = st.selectbox(label='Select class of customer', options=['Working Class', 'Wealthy'])
        Remote_Location = st.selectbox(label='Is customer location Remote?', options=['Yes', 'No'])
        predicted_price = None
        if st.button("Predict"):
               data_dict = CustomData(Artist_Reputation, Height, Material, Price_Of_Sculpture, Base_Shipping_Price,
                              International, Express_Shipment, Installation_Included, Transport,
                              Fragile, Customer_Information, Remote_Location)
               mapped_data = data_dict.get_dataframe()  # Get the mapped data as a dictionary

        # Perform prediction using PredictionPipeline
               prediction_instance = PredictionPipeline()
               input_data = pd.DataFrame(mapped_data, index=[0])  # Create a DataFrame from the mapped data dictionary
               predicted_price = prediction_instance.prrdict(input_data)
            
               st.write(f'The estimated price will be {predicted_price[0]}')


        return predicted_price
            
    def run(self):
        st.sidebar.title("Navigation")
        selected_page = st.sidebar.radio("Go to", list(self.pages.keys()))
        if selected_page == "Home":
            self.home()
        elif selected_page == "Batch Prediction":
            self.batch_prediction()
        elif selected_page == "Train":
            self.train()
        elif selected_page == "Single Prediction":
            self.single_prediction()

    
if __name__ == "__main__":
    app = App()
    app.run()


        

