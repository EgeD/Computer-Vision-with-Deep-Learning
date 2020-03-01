from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg, regtype='L2'):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    - regtype: Regularization type: L1 or L2

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    X_train_len = X.shape[0]
    class_len = W.shape[1]
    ##Take Square
    if regtype == 'L2':
        for i in range(X_train_len):
            score = X[i].dot(W)
            score -= np.max(score)
            denominator = np.sum(np.exp(score))

            loss += -np.log(np.exp(score[y[i]]) / denominator)

            for j in range(class_len):
                grad_k = np.exp(score[j]) / denominator
                dW[:,j] += (grad_k - (j == y[i])) * X[i]

        loss = (loss/X_train_len) + reg * np.sum(W*W)
        dW = (dW/X_train_len) + reg * W
    ##Take Abs
    elif regtype == 'L1':
        for i in range(X_train_len):
            score = X[i].dot(W)
            score -= np.max(score)
            denominator = np.sum(np.exp(score))

            loss += -np.log(np.exp(score[y[i]]) / denominator)

            for j in range(class_len):
                grad_k = np.exp(score[j]) / denominator
                dW[:, j] += (grad_k - (j == y[i])) * X[i]

        loss = (loss/ X_train_len) + reg * np.sum(np.abs(W))
        dW = (dW/X_train_len) + reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, regtype='L2'):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization! Implement both L1 and L2 regularization based on the      #
    # parameter regtype.                                                        #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    if regtype == 'L2':

        X_train_len = X.shape[0]
        scores = X.dot(W)
        scores -= np.max(scores, axis=1, keepdims=True)
        denominator = np.sum(np.exp(scores), axis=1, keepdims=True)
        p = np.exp(scores) / denominator

        loss = np.sum(-np.log(p[np.arange(X_train_len), y]))

        ind = np.zeros_like(p)
        ind[np.arange(X_train_len), y] = 1
        dW = X.T.dot(p - ind)

        loss = (loss / X_train_len) + reg * np.sum(W * W)
        dW = (dW / X_train_len) + reg * W

    elif regtype == 'L1':

        X_train_len = X.shape[0]
        scores = X.dot(W)
        scores -= np.max(scores, axis=1, keepdims=True)
        denominator = np.sum(np.exp(scores), axis=1, keepdims=True)
        p = np.exp(scores) / denominator

        loss = np.sum(-np.log(p[np.arange(X_train_len), y]))

        ind = np.zeros_like(p)
        ind[np.arange(X_train_len), y] = 1
        dW = X.T.dot(p - ind)

        loss = (loss / X_train_len) + reg * np.sum(np.abs(W))
        dW = (dW / X_train_len) + reg * W




    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
