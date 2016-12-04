import random
import numpy as np
from algorithms.data_utils import load_CIFAR10
from algorithms.classifiers import loss_grad_svm_vectorized
from algorithms.gradient_check import grad_check_sparse
from algorithms.classifiers import SVM
import time
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def get_CIFAR10_data(num_training=49000, num_val=1000, num_test=10000, show_sample=True):
    """
    Load the CIFAR-10 dataset, and divide the sample into training set, validation set and test set
    """

    cifar10_dir = '../../DataSets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # subsample the data for validation set
    mask = range(num_training, num_training + num_val)
    X_val = X_train[mask]
    y_val = y_train[list(mask)]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    return X_train, y_train, X_val, y_val, X_test, y_test
    
def preprocessing_CIFAR10_data(X_train, y_train, X_val, y_val, X_test, y_test):
    
    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1)) # [49000, 3072]
    X_val = np.reshape(X_val, (X_val.shape[0], -1)) # [1000, 3072]
    X_test = np.reshape(X_test, (X_test.shape[0], -1)) # [10000, 3072]
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis = 0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))]).T
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))]).T
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))]).T
    return X_train, y_train, X_val, y_val, X_test, y_test


X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw = get_CIFAR10_data()
X_train, y_train, X_val, y_val, X_test, y_test = preprocessing_CIFAR10_data(X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw)

print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape

W = np.random.randn(10, X_train.shape[0]) * 0.001

tic = time.time()
loss_vec, grad_vect = loss_grad_svm_vectorized(W, X_train, y_train, 0)
toc = time.time()
print 'Vectorized loss: %f,computed in %fs' % (loss_vec, toc - tic)
print grad_vect

f = lambda w: loss_grad_svm_vectorized(w, X_train, y_train, 0)[0]
grad_numerical = grad_check_sparse(f, W, grad_vect, 10)
print grad_numerical

SVM_sgd = SVM()
tic = time.time()
losses_sgd = SVM_sgd.train(X_train, y_train, method='sgd', batch_size=200, learning_rate=1e-6,
              reg = 1e5, num_iters=1000, verbose=True, vectorized=True)
toc = time.time()
print 'Traning time for SGD with vectorized version is %f \n' % (toc - tic)

y_train_pred_sgd = SVM_sgd.predict(X_train)[0]
print 'Training accuracy: %f' % (np.mean(y_train == y_train_pred_sgd))
y_val_pred_sgd = SVM_sgd.predict(X_val)[0]
print 'Validation accuracy: %f' % (np.mean(y_val == y_val_pred_sgd))

learning_rates = [1e-5, 1e-8]
regularization_strengths = [10e2, 10e4]

results = {}
best_val = -1
best_svm = None

tic=time.time()
i = 0
interval = 5
for learning_rate in np.linspace(learning_rates[0], learning_rates[1], num=interval):
    i += 1
    print 'The current iteration is %d/%d' % (i, interval)
    for reg in np.linspace(regularization_strengths[0], regularization_strengths[1], num=interval):
        svm = SVM()
        svm.train(X_train, y_train, method='sgd', batch_size=200, learning_rate=learning_rate,
              reg = reg, num_iters=1000, verbose=False, vectorized=True)
        y_train_pred = svm.predict(X_train)[0]
        y_val_pred = svm.predict(X_val)[0]
        train_accuracy = np.mean(y_train == y_train_pred)
        val_accuracy = np.mean(y_val == y_val_pred)
        results[(learning_rate, reg)] = (train_accuracy, val_accuracy)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm
        else:
            pass

for learning_rate, reg in sorted(results):
    train_accuracy,val_accuracy = results[(learning_rate, reg)]
    print 'learning rate %e and regularization %e, \n \
    the training accuracy is: %f and validation accuracy is: %f.\n' % (learning_rate, reg, train_accuracy, val_accuracy)
toc=time.time()

print "Time taken for batch gradient decent to converge is "+str(toc-tic)+" seconds"

y_test_predict_result = best_svm.predict(X_test)
y_test_predict = y_test_predict_result[0]
test_accuracy = np.mean(y_test == y_test_predict)
print 'The test accuracy is: %f' % test_accuracy
