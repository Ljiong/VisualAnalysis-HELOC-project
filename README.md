# HELOC Analysis system

This is a demo of the Dash interactive Python framework developed by [Plotly](https://plot.ly/).

Our motivation is to help understand the model’s inner workings and meet the varied needs through visualization. We focus on an anonymized dataset of Home Equity Line of Credit (HELOC) applications made by real homeowners, containing information about the customers and the result of bad and good performance on their credit accounts.

This system is for the applicants who need an explanation to interpret the HELOC result. The customers do not have any knowledge about the machine learning model, however, they may wonder about the result of their application and how to improve their application if they get a bad result. The main goal of our visualization tool is to help the applicants simulate their application and analysis their performance so that they can use the analysis to improve their performance. Since the random forest model itself is a black-box operation, we need to interpret the model’s results and then provide the corresponding analysis services to the customers of loans in a more straightforward way. 

We designed 4 parts in the system: Individual analysis(subscaled feature & feature analysis), Case-based explanation, Model explanations(feature importance by results & overall importances) and Appendix.

* Our work is inspired by FICO challenge. Feel free to play around!

## Getting Started

### Running the app locally

First create a virtual environment with conda or venv inside a temp folder, then activate it.

```
virtualenv venv

# Windows
venv\Scripts\activate
# Or Linux
source venv/bin/activate


Run the app

```

python app.py

```

## Built With

- [Dash](https://dash.plot.ly/) - Main server and interactive components
- [Plotly Python](https://plot.ly/python/) - Used to create the interactive plots

