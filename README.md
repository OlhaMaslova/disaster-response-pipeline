# Disaster Response Pipeline Project

## Motivation:

Apply software and data engineering skills to analyze disaster data to build a model for an API that classifies disaster messages. Using the web app an emergency worker can input a new message and get classification result in several categories such as "water", "fire", "food", etc.

## Data:

The dataset was provided by Figure Eight and contained real messages that were sent during disaster events.

## Final Result:

Build a web app that will prompt users to enter a message and return corresponding category predictions based on the ML pipeline.

## Instructions:

1. Install dependencies in your virtual environment by running:

   `conda create --name <env> --file <requirements.txt>`

2. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

     Note: initial model training may take a few hours depending on your machine.

3. Run the following command in the app's directory to run your web app.
   `python run.py`

4. Go to http://0.0.0.0:3001/
