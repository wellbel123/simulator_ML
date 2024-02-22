import numpy as np

'''
    Task 1: a forecast of the turnover of household appliances.
    it's necessary to understand  how much of each product to bring to a particular point for sale
    Which is preferable in this situation: 
    to bring more than necessary to the warehouse or less?
    You need to come up with a loss function that would be adequate for this business task

    The decision: In this situation, it would be preferable to bring more goods and work 
    on the turnover rate, rather than bring less goods,
    so the function will penalize more for under-forcasting. 
    Depending on the data, we can select the coefficient which can be < 1 (for under-forecasting) and >1 (for over-forecasting)
'''


def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    error = np.mean(((y_true - y_pred) / y_pred) ** 2)
    return error


'''
    Task 2: Lifetime Value Rating 
    We are a B2B fintech startup – we provide deposits/loans, purchase/sale of securities and other financial instruments.
    We have a small number of very large clients. We conclude contracts for at least 1 year and to make a decision on cooperation,
    we develop a model that predicts LTV for a potential client.
    Think about which is preferable: to underestimate or overestimate the value of a potential client?
    You need to select or develop an error-reflecting function to evaluate the LTV prediction model.

    The decision: we will also use MSE and will penalize for over-forecasting
'''


def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    error = np.mean(((y_true - y_pred) * y_pred)**2)
    return error
