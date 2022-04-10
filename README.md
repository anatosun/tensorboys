# Pattern Recognition on MNIST dataset

## Support Vector Machine (SVM)

### Average accuracy and Best parameters

LINEAR(C=0.01, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=1.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=10.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=100.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=1000.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=10000.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=100000.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=1000000.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=10000000.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=100000000.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=1000000000.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=10000000000.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=100000000000.0, gamma=1e-09): Average accuracy was: 0.9106  
LINEAR(C=1000000000000.0, gamma=1e-09): Average accuracy was: 0.9106  

LINEAR: The best parameters are {'C': 0.01, 'gamma': 1e-09} with a score of 0.91060 


RBF(C=1000.0, gamma=1e-09): Average accuracy was: 0.9234  
RBF(C=1000.0, gamma=1e-08): Average accuracy was: 0.9258000000000001  
RBF(C=1000.0, gamma=1e-07): Average accuracy was: 0.9536  
RBF(C=1000.0, gamma=1e-06): Average accuracy was: 0.9338  
RBF(C=1000.0, gamma=1e-05): Average accuracy was: 0.1774  
RBF(C=1000.0, gamma=0.0001): Average accuracy was: 0.1112  
RBF(C=1000.0, gamma=0.001): Average accuracy was: 0.1112  
RBF(C=1000.0, gamma=0.01): Average accuracy was: 0.1112  
RBF(C=1000.0, gamma=0.1): Average accuracy was: 0.1112  
RBF(C=1000.0, gamma=1.0): Average accuracy was: 0.1112  
RBF(C=1000.0, gamma=10.0): Average accuracy was: 0.1112  
RBF(C=1000.0, gamma=100.0): Average accuracy was: 0.1112  
RBF(C=1000.0, gamma=1000.0): Average accuracy was: 0.1112  
RBF(C=1000.0, gamma=10000.0): Average accuracy was: 0.1112  
RBF(C=1000.0, gamma=100000.0): Average accuracy was: 0.1112  
RBF(C=10000.0, gamma=1e-09): Average accuracy was: 0.9166000000000001  
RBF(C=10000.0, gamma=1e-08): Average accuracy was: 0.9258000000000001  
RBF(C=10000.0, gamma=1e-07): Average accuracy was: 0.9536  
RBF(C=10000.0, gamma=1e-06): Average accuracy was: 0.9338  
RBF(C=10000.0, gamma=1e-05): Average accuracy was: 0.1774  
RBF(C=10000.0, gamma=0.0001): Average accuracy was: 0.1112  
RBF(C=10000.0, gamma=0.001): Average accuracy was: 0.1112  
RBF(C=10000.0, gamma=0.01): Average accuracy was: 0.1112  
RBF(C=10000.0, gamma=0.1): Average accuracy was: 0.1112  
RBF(C=10000.0, gamma=1.0): Average accuracy was: 0.1112  
RBF(C=10000.0, gamma=10.0): Average accuracy was: 0.1112  
RBF(C=10000.0, gamma=100.0): Average accuracy was: 0.1112  
RBF(C=10000.0, gamma=1000.0): Average accuracy was: 0.1112  
RBF(C=10000.0, gamma=10000.0): Average accuracy was: 0.1112  
RBF(C=10000.0, gamma=100000.0): Average accuracy was: 0.1112  
RBF(C=100000.0, gamma=1e-09): Average accuracy was: 0.9166000000000001  
RBF(C=100000.0, gamma=1e-08): Average accuracy was: 0.9258000000000001  
RBF(C=100000.0, gamma=1e-07): Average accuracy was: 0.9536  
RBF(C=100000.0, gamma=1e-06): Average accuracy was: 0.9338  
RBF(C=100000.0, gamma=1e-05): Average accuracy was: 0.1774  
RBF(C=100000.0, gamma=0.0001): Average accuracy was: 0.1112  
RBF(C=100000.0, gamma=0.001): Average accuracy was: 0.1112  

RBF: The best parameters are {'C': 100.0, 'gamma': 1e-07} with a score of 0.95360 

All the averages are in the file "SVM_crossvalidation_averages.txt"

### Accuracy

Linear: 0.902
RBF: 0.944

## Multi-layer Perceptron (MLP)

### Accuracy with respect to the training epochs

![mlp_epoch_plot](https://user-images.githubusercontent.com/85929824/162423918-3420180e-6c95-4213-b52a-5f8907a31564.png)

The first figure shows that the loss decrease dramatically with the number of epochs and the second figure shows that the accuracy increases quickly at first before reaching a plateau when the number of epochs grows.

### Accuracy with respect to the architecture

![mlp_accuracy](https://github.com/fwicht/mnist-recognition/blob/main/img/mlp_archi.png)

This table shows the different architectures sorted according to their accuracy. The following figure shows that the accuracy increases dramatically with the number of parameters but plateau quickly when there are more than 100 000 parameters.

![mlp_accuracy_plot](https://user-images.githubusercontent.com/85929824/162430244-95342d2f-4fbd-434d-a343-9e7734e50198.png)

Here we can see the zoomed section, with more details. Ideally, the line should be as close as possible to a straight angle. We can see that with 3 or 4 layers, we get a better precision for the same number of parameters.

![mlp_zoomed](https://github.com/fwicht/mnist-recognition/blob/main/img/mlp_layers.png)

### Conclusion

The more layers we add the less the model is sensitive to parameters addition. In other words, if we have many layers, increasing their size has very few effects. On the other hand, with few layers, increasing their size has a strong effect on accuracy. We therefore find out that we already got a pretty good accuracy with only one dense layer of 512 entries. The optimal network size seems to be the one with 4 layers and the architecture 784x512x128x64x10 that sums up to 125898 parameters.

## Convolutional Neural Network (CNN)

We played a bit with different architectures.

xL is the number of convolutional layers, the simple is a traditional architecture. For the dropout, we added a dropout layer after the conv, and the kernel is with a bigger kernel of size (5,5) instead of the size (3,3)

![cnn_archi](https://github.com/fwicht/mnist-recognition/blob/main/img/cnn_perf.png)


And finally here is the impact of the epochs
The test set was used as validation test during the training. Obviously it is bad if we want to compute the score over the test set at the end. But it is an easy trick to see the precision of the model on both the training and the test set after every epochs

![cnn_epoch](https://github.com/fwicht/mnist-recognition/blob/main/img/cnn_epochs.png)