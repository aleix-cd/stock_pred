# Project: Microsoft's Stock Price Prediction

#### Author

[Aleix Cubel Capdevila](https://www.linkedin.com/in/aleix-cubel/)

## Objective

The goal of this project is to develop a regression model with a Long Short-Term Memory network that is able to forecast the price of Microsoft's Stock looking solely at the last 21 days of data.

Effectively predicting the price of stocks has its obvious useful applications, but it is important not to forget its stochastic nature.


## Table of Contents

1. **Data Extraction**
2. **Feature Engineering**
3. **Exploratory Data Analysis**
4. **Statistical Checks**
5. **Feature Selection**
6. **Modelling and Hyperparameter Optimization**


### 1. Data Extraction

As the very first part of this project, this section is developed to extract the data from different sources and prepare it for its analysis.

#### Stock Prices: Microsoft, Correlated Assets, Currency Pairs and Indexes

[Yahoo Finance API](https://pypi.org/project/yfinance/) was the tool I used to extract the historical data of all the stocks. Being easy to use, only had to specify the tickers and the start/end date to pull all the data.

#### Sentiment Analysis: finBERT

I wanted to use natural language processing, text analysis and computational linguistics in order to extract and identify subjective information on worldwide news. This way, my model would technically be able to predict the behavior and trend of the stock market.

To do so, I used the [finBERT project](https://github.com/ProsusAI/finBERT), which was specifically made to perform sentiment analysis of financial texts, since it uses a large financial corpus and it has been therefore fine-tuned for financial sentiment classification.

### 2. Feature Engineering

For trend analysis purpose, I computed 

- **Moving Averages (7 and 21 days)**
    - These help reduce the noise from random short-term price fluctuation. 
    - We use them to determine the trend direction and resistance levels.
- **Moving Average Convergence Divergence (MACD)**
    - Shows the relationship between two moving averages.
    - Helps identify bullish/bearish movements, and their intensity.
- **Bollinger Bands**
    - Formed by the upper, the middle and the lower band, these help generate oversold/overbought signals.
- **Expontential Moving Average**
    - Moving average that gives more weight and importance to the most recent data points.
- **Returns**
    - Simple indicator that computes the percentage of change of the price in relation with the prior day.
    - Helps indentifying trends.
    
**Note:** due to their high dependancy on the target variable, many of them were dropped before modelling and were used merely for visualization purposes.

### 3. Exploratory Data Analysis

Performed a complete EDA by visualizing the distributions, applying transformations and comparing the several features available with correlations matrices.


### 4. Statistical Checks

- Heteroscedasticity: 

`What you have in your data when the conditional variance is not constant, understanding conditional variance as the variability that you see in y (dependant variable) for each value of t (time period).`

Given the nature of most of the variables in the dataset (stock prices, indexes and currency pairs), and after seeing the plots in the EDA section, we can confirm that most of the features are nonlinear and heteroscedastic.

- Multicollinearity:

`Multicollinearity is given when two or more independent variables are highly correlated with one another in a regression model`.

The Variance Inflation Factor (VIF) will easily let us see the multicollinearity of X. VIF starts at 1 and has no upper limit. 

- VIF = 1, no correlation between the independent variable and the other variables
- VIF > 20, high multicollinearity between this independent variable and the others


- Autocorrelation

`Autocorrelation is a mathematical representation of the degree of similarity between a given time series and a lagged version of itself over successive time intervals.` [[Source]](https://www.investopedia.com/terms/a/autocorrelation.asp#:~:text=Autocorrelation%20represents%20the%20degree%20of,itself%20over%20successive%20time%20intervals.&text=An%20autocorrelation%20of%20%2B1%20represents%20a%20perfect%20positive%20correlation%2C%20while,represents%20a%20perfect%20negative%20correlation.)


### 5. Feature Selection

By using a tree-based algorithm,  we are able to see the importance of each of the features when a decision is made. I decided, then, to use a pretty popular algorithm, XGBoost, in order to see what features the model was taking more into account.

Figure: Example of feature selection plot with Indexes
![random text](https://i.imgur.com/TgBC7pa.png)


### 6. Modelling and Hyperparameter Optimization

I had clear that I wanted to use **Long Short-Term Memory Networks**, due to their sequence nature and therefore good performance on time series.

The architectures I chose are:
- Univariate Vanilla-LSTM (baseline)
- Multivariate Vanilla-LSTM
- Multivariate Stacked-LSTM
- Multivariate CNN-LSTM
- Bidirectional-LSTM

For hyperparameter tuning and optimization, I used `hyperopt` with `EarlyStopping callbacks`, in order to iterate over different architectures and test them for a few epochs each.


## Results

**Univariate Vanilla-LSTM (baseline model)**

MSE (test): 0.4053

[Imgur](https://imgur.com/HwHMj7X)

**Multivariate Vanilla-LSTM**

MSE (test): 0.3570

[Imgur](https://imgur.com/J0Glu29)

**Multivariate Stacked-LSTM**

MSE (test): 1.2658

[Imgur](https://imgur.com/undefined)

**Multivariate CNN-LSTM**

MSE (test): 0.9602

[Imgur](https://imgur.com/uMRjwho)

**Bidirectional-LSTM**

MSE (test): 0.3985

[Imgur](https://imgur.com/mlKp6AO)

**Autoregressive Integrated Moving Average (ARIMA)**

MSE (test): 0.517 

[Imgur](https://imgur.com/2FMvwBx)


## Conclusions

This project reported the results of experimentation, through which the performance and accuracy as well as behavioral training of Vanilla/Stacked LSTM, CNN-LSTM and Bidirectional-LSTM models were analyzed, hyper-tuned and compared. I also added a forecast made with an ARIMA model.

The main objective of this project was primarily to focus on whether adding correlated assets and sentiment analysis would help on improving the precision of time series forecasting applied to the stock market, and compare the results between univariate and multivariate models.

The results, even though are not 100% conclusive, state that adding more data - such as the one mentioned above - for stock price forecasting can easily incur in model confusion and poor generalization on unseen data. 

Nevertheless, simpler models such as the Vanilla-LSTM (multivariate) and the Bi-LSTM performed extremely well, the latter showing exceptional results at a higher computational cost.

In relation with the Bi-LSTM, I noticed that the training based on it is way slower since it takes additional batches of data. This could explain both the -better- loss and error, which in my opinion outperforms the rest of architectures, and could indicate that there are additional features captured by the bidirectionality that are not taken into account on unidirectional models.
