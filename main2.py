import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes

import torch
import time
import matplotlib.pyplot as plt
from csv import writer

def main(args):
    """
    The main function of the script. Do not hesitate to play with it
    and add your own code, visualization, prints, etc!

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end 
                          of this file). Their value can be accessed as "args.argument".
    """
    ## 1. First, we load our data and flatten the images into vectors
    xtrain, xtest, ytrain, ytest = load_data(args.data)
    xtrain = xtrain.reshape(xtrain.shape[0], -1)
    xtest = xtest.reshape(xtest.shape[0], -1)



    ## 2. Then we must prepare it. This is were you can create a validation set,
    #  normalize, add bias, etc.
    
    # shuffling data
    np.random.seed(0)
    rinds = np.random.permutation(len(xtrain))  # shuffling of the indices to shuffle the data
    
    # Make a validation set
    if not args.test:
        #make new xtest as some of xtrain, same for y
        fraction_train = 0.8  # 80% of data is reserved for training, so 20% for testing
        n_train = int(len(xtrain) * fraction_train)
        xtest = xtrain[rinds[n_train:]] 
        ytest = ytrain[rinds[n_train:]] 
        xtrain = xtrain[rinds[:n_train]] 
        ytrain = ytrain[rinds[:n_train]] 
        iftest = 0
    else:
        xtrain = xtrain[rinds]
        ytrain = ytrain[rinds]
        iftest = 1
    
    ### WRITE YOUR CODE HERE to do any other data processing
    #normalise
    meantrain = xtrain.mean(0,keepdims=True)
    stdtrain  = xtrain.std(0,keepdims=True)
    xtrain = normalize_fn(xtrain,meantrain,stdtrain)
    xtest = normalize_fn(xtest,meantrain,stdtrain)

    # Dimensionality reduction (MS2)
    if args.use_pca:
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data


    ## 3. Initialize the method you want to use.

    # Neural Networks (MS2)
    if args.method == "nn":
        print("Using deep network")

        # Prepare the model (and data) for Pytorch
        # Note: you might need to reshape the image data depending on the network you use!
        n_classes = get_n_classes(ytrain)
        if args.nn_type == "mlp":
            torch.manual_seed(42)
            model = MLP( len(xtrain[0]), n_classes)  ### WRITE YOUR CODE HERE


        elif args.nn_type == "cnn":
            ### WRITE YOUR CODE HERE
            # Reshape the image data for convolutional layers
            height = 32
            width = 32
            
    
            xtrain = xtrain.reshape(xtrain.shape[0], 1, height, width)
            xtest = xtest.reshape(xtest.shape[0], 1, height, width) 


            model = CNN(input_channels=1, n_classes=n_classes)
        
        summary(model)

        # Trainer object
        method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
    
    # Follow the "DummyClassifier" example for your methods (MS1)
    elif args.method == "dummy_classifier":
        method_obj =  DummyClassifier(arg1=1, arg2=2)
    

    ## 4. Train and evaluate the method

    # Fit (:=train) the method on the training data
    s1 = time.time()
    preds_train = method_obj.fit(xtrain, ytrain)
    s2 = time.time()
    print('')
    train_time = s2 - s1
    print("Training Time: ", train_time, " seconds.")

    # Predict on unseen data
    s1 = time.time()
    preds = method_obj.predict(xtest)
    s2 = time.time()
    test_time = s2 - s1
    print("Testing Time: ", test_time, " seconds.")


    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    train_macrof1 = macrof1_fn(preds_train, ytrain)
    d1 = [acc/100.0,train_macrof1]
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {train_macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    test_macrof1 = macrof1_fn(preds, ytest)
    d2 = [acc/100.0,test_macrof1]
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {test_macrof1:.6f}")


    if args.method == "nn" and args.nn_type == "mlp":
        with open('results_mlp.csv', 'a') as f_object:
            applist = [1, [64], args.lr, args.max_iters, train_macrof1, test_macrof1, iftest, train_time, test_time ]
            writer_object = writer(f_object)
            writer_object.writerow(applist)
            f_object.close()

    ## WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.
    barWidth = 0.25
    fig = plt.subplots(figsize =(6, 6))
    br1 = np.arange(len(d1))
    br2 = [x + barWidth for x in br1]
    plt.bar(br1, d1, color ='khaki', width = barWidth,
            edgecolor ='grey', label ='Train set')
    plt.bar(br2, d2, color ='deepskyblue', width = barWidth,
            edgecolor ='grey', label ='Test set')
    plt.ylim(0,1)
    plt.title('Model performance')
    plt.ylabel('Score', fontweight ='bold', fontsize = 15)
    plt.xticks([r + barWidth for r in range(len(d1))],
            ['Accuracy', 'Macro F1 Score',])
    plt.legend()
    #plt.show()


if __name__ == '__main__':
    # Definition of the arguments that can be given through the command line (terminal).
    # If an argument is not given, it will take its default value as defined below.
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset_HASYv2", type=str, help="the path to wherever you put the data, if it's in the parent folder, you can use ../dataset_HASYv2")
    parser.add_argument('--method', default="dummy_classifier", type=str, help="dummy_classifier / kmeans / logistic_regression / svm / nn (MS2)")
    parser.add_argument('--K', type=int, default=10, help="number of clusters for K-Means")
    parser.add_argument('--lr', type=float, default=1e-5, help="learning rate for methods with learning rate")
    parser.add_argument('--max_iters', type=int, default=100, help="max iters for methods which are iterative")
    parser.add_argument('--test', action="store_true", help="train on whole training data and evaluate on the test data, otherwise use a validation set")
    parser.add_argument('--svm_c', type=float, default=1., help="Constant C in SVM method")
    parser.add_argument('--svm_kernel', default="linear", help="kernel in SVM method, can be 'linear' or 'rbf' or 'poly'(polynomial)")
    parser.add_argument('--svm_gamma', type=float, default=1., help="gamma prameter in rbf/polynomial SVM method")
    parser.add_argument('--svm_degree', type=int, default=1, help="degree in polynomial SVM method")
    parser.add_argument('--svm_coef0', type=float, default=0., help="coef0 in polynomial SVM method")

    ### WRITE YOUR CODE HERE: feel free to add more arguments here if you need!

    # MS2 arguments
    parser.add_argument('--use_pca', action="store_true", help="to enable PCA")
    parser.add_argument('--pca_d', type=int, default=200, help="output dimensionality after PCA")
    parser.add_argument('--nn_type', default="mlp", help="which network to use, can be 'mlp' or 'cnn'")
    parser.add_argument('--nn_batch_size', type=int, default=64, help="batch size for NN training")

    # "args" will keep in memory the arguments and their values,
    # which can be accessed as "args.data", for example.
    args = parser.parse_args()
    main(args)
