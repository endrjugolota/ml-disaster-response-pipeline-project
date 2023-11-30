# Disaster Response Pipeline Project


### Overview
Aim of this project is to create an API that classifies disaster messages.

Data used for this project are disaster messages from Appen (formally Figure 8).

Original data set used contains real messages that were sent during disaster events.

This data is used to create a machine learning pipeline to categorize these events in 36 categories:
related, request, offer, aid_related, medical_help, medical_products, search_and_rescue, security, military, child_alone, water, food, shelter, clothing, money, missing_people, refugees, death, other_aid, infrastructure_related, transport, buildings, electricity, tools, hospitals, shops, aid_centers, other_infrastructure, weather_related, floods, storm, fire, earthquake, cold, other_weather, direct_report

The code in this repository:
- runs an ETL pipeline which loads, cleans and saves the data into sqlite database
- runs a ML pipline on preapared data to create model for classification messages
- outputs model as a picle file
- creates web app where an one can input a new message and get classification results in several categories

The web app also displays visualizations of the data.


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to page specified in the console.


### Code structure
- app
| - template 
| |- master.html  # main page of web app
| |- go.html      # classification result page of web app
|- run.py         # Flask file that runs app

- data
|- disaster_categories.csv  # data to process 
|- disaster_messages.csv    # data to process
|- process_data.py          # python script for cleaning and transforming the data 
|- DisasterResponse.db      # database to save clean data to

- models
|- train_classifier.py     # python script creating, training and evaluating ml model  
|- classifier.pkl          # saved model 
