# Disaster Response Pipeline Project

### Introduction

This repository is the work for my project from the Udacity Data Scientist Nanodegree Program. I used the Disaster data from [Appen](https://appen.com/) (formally Figure 8) containing real messages that were sent during disaster events to build a model for an API that classifies disaster messages. The classification model will help people from disaster organizations classify the message into related categories so they can respond to the event more accurately and faster.

In this project, I applied data engineering skills to build an ETL pipeline to process the raw data, the data then will go through an ML pipeline to classify data. My project also includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app will also display visualizations of the data.

### Structure

```bash
├── app
│   ├── run.py # Flask file that runs app
│   └── template
│       ├── go.html # classification result page of web app
│       └── master.html # main page of web app
├── data
│   ├── DisasterResponse.db # database to save clean data
│   ├── disaster_categories.csv # data to process
│   ├── disaster_messages.csv # data to process
│   └── process_data.py
├── models
│   └── train_classifier.py
│   └── classifier.pkl # saved model
├── README.md
```

### Instructions:

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python run.py`

3. Go to http://0.0.0.0:3001/
