# Random Forest Classification Model to Predict Stock Prices

This repository contains a Python implementation of a Random Forest classification model that predicts the direction of stock prices (up or down) using the historical prices of the S&P 500 index (^GSPC).

## Getting Started

These instructions will help you run the code on your local machine.

## Prerequisites

The following libraries are required to run the code:

- numpy
- pandas
- yfinance
- matplotlib
- pandas_datareader
- sklearn

## Installation

To install the required libraries, run the following command in your terminal:

```
pip install numpy pandas yfinance matplotlib pandas_datareader scikit-learn
```

## Usage

1. Clone the repository.
2. Open the **'predict_stock_prices.ipynb'** file in Jupyter Notebook.
3. Run the code cells to execute the model.
4. The output will be displayed in the Notebook.

## Methodology

1. Download the data using the **'yfinance'** library.
2. Calculate the returns and logarithmic returns of the data.
3. Create independent variables using the historical prices of the stock market, exponential moving averages, conditionals, moving average convergence divergence, relative strength index, stochastic oscillator, and rate of change.
4. Create dependent variables that represent whether the returns of the next day are positive or negative and whether the price of the stock is going to be higher or lower than the previous day.
5. Split the data into training and testing sets using the **'train_test_split'** function from the **'sklearn'** library.
6. Train the model using the **'RandomForestClassifier'** class from the **'sklearn'** library.
7. Test the model on the testing data.
8. Evaluate the performance of the model using the **'classification_report'** and ***'accuracy_score'** functions from the **'sklearn'** library.
