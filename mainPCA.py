#main file to test PCA on methods from MS1
#key explanations are preceded by the comment: "MS2 PCA"
import argparse

import numpy as np
from torchinfo import summary

from src.data import load_data
from src.methods.dummy_methods import DummyClassifier
from src.methods.pca import PCA
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, get_n_classes
from src.methods.kmeans import KMeans
from src.methods.logistic_regression import LogisticRegression
from src.methods.svm import SVM
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn
import time
import torch
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
    else:
        xtrain = xtrain[rinds]
        ytrain = ytrain[rinds]
    
    ### WRITE YOUR CODE HERE to do any other data processing
    #normalise
    meantrain = xtrain.mean(0,keepdims=True)
    stdtrain  = xtrain.std(0,keepdims=True)
    xtrain = normalize_fn(xtrain,meantrain,stdtrain)
    xtest = normalize_fn(xtest,meantrain,stdtrain)

    # Dimensionality reduction (MS2)
    if args.use_pca:
        # MS2 PCA:
        # First perform prediction without PCA, measure: prediction time, accuracy on test set, f1 on test set for comparison
        accuracyplttrain = []
        f1plttrain = []
        accuracyplttest = []
        f1plttest = []
        dplt = []
        exvarplt = []
        comparepca_acc = [] 
        comparepca_f1 = []
        comparepca_time = []
        timeplt = []
        print("without PCA")
        n_classes = get_n_classes(ytrain)
        if args.method == "nn":
            if args.nn_type == "mlp":
                torch.manual_seed(42)
                model = MLP( len(xtrain[0]), n_classes)  ### WRITE YOUR CODE HERE
                summary(model)
                method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
            else:
                pass
        elif args.method == "logistic_regression": 
            #add bias
            append_bias_term(xtrain)
            append_bias_term(xtest)
            method_obj =  LogisticRegression(lr = args.lr, max_iters = args.max_iters)
        elif args.method == "kmeans": 
            method_obj =  KMeans(K = args.K)
        elif args.method == "svm": 
            if args.svm_degree and args.svm_coef0:
                method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, degree = args.svm_degree, coef0 = args.svm_coef0)
            elif args.svm_degree:
                method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, degree = args.svm_degree)
            elif args.svm_coef0:
                method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, coef0 = args.svm_coef0)
            else:
                method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma)
        else:
            pass
        
        preds_train = method_obj.fit(xtrain, ytrain)

        # Predict on unseen data
        start = time.time()
        preds = method_obj.predict(xtest)
        end = time.time()
        comparepca_time.append(end-start)

        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        comparepca_acc.append(acc)
        comparepca_f1.append(macrof1)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        # MS2 PCA:
        # Perform prediction with PCA with specified d value, measure: prediction time, accuracy on test set, f1 on test set for comparison
        print("Using PCA")
        pca_obj = PCA(d=args.pca_d)#25
        ### WRITE YOUR CODE HERE: use the PCA object to reduce the dimensionality of the data
        exvar = pca_obj.find_principal_components(xtrain)
        print("d: ", pca_obj.d)
        dplt.append(pca_obj.d)
        print("Explained variance: ", exvar)
        exvarplt.append(exvar)
        xtrain = pca_obj.reduce_dimension(xtrain)
        xtest = pca_obj.reduce_dimension(xtest)
        print("Using deep network")

        # Prepare the model (and data) for Pytorch
        # Note: you might need to reshape the image data depending on the network you use!
        n_classes = get_n_classes(ytrain)
        if args.method == "nn":
            if args.nn_type == "mlp":
                torch.manual_seed(42)
                model = MLP( len(xtrain[0]), n_classes)  ### WRITE YOUR CODE HERE
                summary(model)
                method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
            else:
                pass
        elif args.method == "logistic_regression": 
            #add bias
            append_bias_term(xtrain)
            append_bias_term(xtest)
            method_obj =  LogisticRegression(lr = args.lr, max_iters = args.max_iters)
        elif args.method == "kmeans": 
            method_obj =  KMeans(K = args.K)
        elif args.method == "svm": 
            if args.svm_degree and args.svm_coef0:
                method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, degree = args.svm_degree, coef0 = args.svm_coef0)
            elif args.svm_degree:
                method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, degree = args.svm_degree)
            elif args.svm_coef0:
                method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, coef0 = args.svm_coef0)
            else:
                method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma)
        else:
            pass
        preds_train = method_obj.fit(xtrain, ytrain)

        # Predict on unseen data
        start = time.time()
        preds = method_obj.predict(xtest)
        end = time.time()
        timeplt.append(end-start)


        ## Report results: performance on train and valid/test sets
        acc = accuracy_fn(preds_train, ytrain)
        macrof1 = macrof1_fn(preds_train, ytrain)
        accuracyplttrain.append(acc)
        f1plttrain.append(macrof1)
        print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        acc = accuracy_fn(preds, ytest)
        macrof1 = macrof1_fn(preds, ytest)
        f1plttest.append(macrof1)
        accuracyplttest.append(acc)
        print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

        # MS2 PCA:
        # Test different values of hyperparameter d to measure the impact on: 
        # Expected variance, Accuracy and F1 for the test set, and Accuracy and F1 for the train set
        for x in range(50,500,50):
            pca_obj = PCA(d=x)
            exvar = pca_obj.find_principal_components(xtrain)
            print("d: ", pca_obj.d)
            dplt.append(pca_obj.d)
            print("Explained variance: ", exvar)
            exvarplt.append(exvar)
            xtrain = pca_obj.reduce_dimension(xtrain)
            xtest = pca_obj.reduce_dimension(xtest)
            #if args.method == "nn":
            print("Using deep network")

            # Prepare the model (and data) for Pytorch
            # Note: you might need to reshape the image data depending on the network you use!
            n_classes = get_n_classes(ytrain)
            if args.method == "nn":
                if args.nn_type == "mlp":
                    torch.manual_seed(42)
                    model = MLP( len(xtrain[0]), n_classes)  ### WRITE YOUR CODE HERE
                    summary(model)
                    method_obj = Trainer(model, lr=args.lr, epochs=args.max_iters, batch_size=args.nn_batch_size)
                else:
                    pass
            if args.method == "logistic_regression": 
            #add bias
                append_bias_term(xtrain)
                append_bias_term(xtest)
                method_obj =  LogisticRegression(lr = args.lr, max_iters = args.max_iters)
            elif args.method == "kmeans": 
                method_obj =  KMeans(K = args.K)
            elif args.method == "svm": 
                if args.svm_degree and args.svm_coef0:
                    method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, degree = args.svm_degree, coef0 = args.svm_coef0)
                elif args.svm_degree:
                    method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, degree = args.svm_degree)
                elif args.svm_coef0:
                    method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma, coef0 = args.svm_coef0)
                else:
                    method_obj =  SVM(C = args.svm_c, kernel = args.svm_kernel, gamma = args.svm_gamma)
            else:
                pass
            
            preds_train = method_obj.fit(xtrain, ytrain)
        
            # Predict on unseen data
            start = time.time()
            preds = method_obj.predict(xtest)
            end = time.time()
            timeplt.append(end-start)


            ## Report results: performance on train and valid/test sets
            acc = accuracy_fn(preds_train, ytrain)
            macrof1 = macrof1_fn(preds_train, ytrain)
            accuracyplttrain.append(acc)
            f1plttrain.append(macrof1)
            print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

            acc = accuracy_fn(preds, ytest)
            macrof1 = macrof1_fn(preds, ytest)
            f1plttest.append(macrof1)
            accuracyplttest.append(acc)
            print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")
        
        #MS2 PCA:
        # Compare time, accuracy and F1 for the test set without PCA vs with PCA with the hyperparameter d that performed with maximum test accuracy
        idxmax = accuracyplttest.index(max(accuracyplttest))
        comparepca_acc.append(accuracyplttest[idxmax])
        comparepca_f1.append(f1plttest[idxmax])
        comparepca_time.append(timeplt[idxmax])
        comparepca_labels = ["without PCA", "with PCA"]
        
        barWidth = 0.25
        fig = plt.subplots(figsize =(12, 6))
        
        d1 = [comparepca_acc[0]/100,comparepca_f1[0]]
        d2 = [comparepca_acc[1]/100,comparepca_f1[1]]
        br1 = np.arange(len(d1))
        br2 = [x + barWidth for x in br1]
        plt.bar(br1, d1, color ='b', width = barWidth,
                edgecolor ='grey', label ='without PCA')
        plt.bar(br2, d2, color ='y', width = barWidth,
                edgecolor ='grey', label ='with PCA')
        plt.ylim(0,1)
        plt.title('Model performance with vs without PCA ')
        plt.ylabel('Fraction', fontweight ='bold', fontsize = 15)
        plt.xticks([r + barWidth for r in range(len(d1))],
                ['Accuracy', 'F1 Score',])
        plt.legend()
        plt.show()
        
        print("comparepca_time[0]", comparepca_time[0])
        print("comparepca_time[1]", comparepca_time[1])
        fig = plt.figure()
        #ax = fig.add_axes([0,0,1,1])
        plt.bar(comparepca_labels,comparepca_time)
        plt.title('Time taken with vs without PCA ')
        #plt.xticks([],['without PCA','with PCA'])
        plt.show()

        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True

        ax = plt.GridSpec(2, 2)
        ax.update(wspace=0.5, hspace=0.5)
        
        # MS2 PCA:
        # Plots to compare the impact of hyperparameter d on Expected variance, and Accuracy and F1 for train and test sets
        
        plt.subplot(2,3,1)
        plt.scatter(dplt,exvarplt,c='r')
        plt.title("d vs Exvar")
        plt.grid()


        plt.subplot(2,3,3)
        plt.scatter(dplt,accuracyplttrain,c='g')
        plt.title("d vs Accuracy (train)")
        plt.grid()


        plt.subplot(2,3,4)
        plt.scatter(dplt,f1plttrain,c='b')
        plt.title("d vs F1 (train)")
        plt.grid()


        plt.subplot(2,3,5)
        plt.scatter(dplt,accuracyplttest,c='y')
        plt.title("d vs Accuracy (test)")
        plt.grid()
        
        plt.subplot(2,3,6)
        plt.scatter(dplt,f1plttest,c='b')
        plt.title("d vs F1 (test)")
        plt.grid()
        
        plt.show()


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
    preds_train = method_obj.fit(xtrain, ytrain)
        
    # Predict on unseen data
    preds = method_obj.predict(xtest)


    ## Report results: performance on train and valid/test sets
    acc = accuracy_fn(preds_train, ytrain)
    macrof1 = macrof1_fn(preds_train, ytrain)
    print(f"\nTrain set: accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    acc = accuracy_fn(preds, ytest)
    macrof1 = macrof1_fn(preds, ytest)
    print(f"Test set:  accuracy = {acc:.3f}% - F1-score = {macrof1:.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.



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
