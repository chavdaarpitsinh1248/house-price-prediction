import numpy as np

def predict(X, W, b):
    return np.dot(X, W) + b


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def mse_l2_loss(y_true, y_pred, W, lambda_):
    mse = np.mean((y_true - y_pred) ** 2)
    l2_penalty = lambda_ * np.sum(W ** 2)
    return mse + l2_penalty


def compute_gradients(X, y, y_pred):
    n = len(y)
    dW = (-2 / n) * np.dot(X.T, (y - y_pred))
    db = (-2 / n) * np.sum(y - y_pred)
    return dW, db


def compute_gradients_l2(X, y, y_pred, W, lambda_):
    n = len(y)
    dW = (-2 / n) * np.dot(X.T, (y - y_pred)) + 2 * lambda_ * W
    db = (-2 / n) * np.sum(y - y_pred)
    return dW, db


def gradient_descent(X, y, W, b, learning_rate=0.01, epochs=1000):
    loss_history = []
    
    for epoch in range(epochs):
        y_pred = predict(X, W, b)
        loss = mean_squared_error(y, y_pred)
        loss_history.append(loss)
        
        dW, db = compute_gradients(X, y, y_pred)
        
        W -= learning_rate * dW
        b -= learning_rate * db
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
    return W, b, loss_history

def gradient_descent_l2(X, y, W, b, learning_rate=0.01, epochs=1000, lambda_=0.1):
    loss_history = []
    for epoch in range(epochs):
        y_pred = predict(X, W, b)
        loss = mse_l2_loss(y, y_pred, W, lambda_)
        loss_history.append(loss)
        
        dW, db = compute_gradients_l2(X, y, y_pred, W, lambda_)
        
        W -= learning_rate * dW
        b -= learning_rate * db
        
    return W, b, loss_history