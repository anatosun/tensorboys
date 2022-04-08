# Pattern Recognition on MNIST dataset

## Support Vector Machine (SVM)

## Multi-layer Perceptron (MLP)

### Accuracy with respect to the training epochs

![mlp_epoch_plot](https://user-images.githubusercontent.com/85929824/162423918-3420180e-6c95-4213-b52a-5f8907a31564.png)

The first figure shows that the loss decrease dramatically with the number of epochs and the second figure shows that the accuracy increases quickly at first before reaching a plateau when the number of epochs grows.

### Accuracy with respect to the architecture

![mlp_accuracy](https://user-images.githubusercontent.com/85929824/162429919-850cf469-0a45-474a-ba46-2da4179134c2.png)

This table shows the different architectures sorted according to their accuracy. The following figure shows that the accuracy increases dramatically with the number of parameters but plateau quickly when there are more than 100 000 parameters.

![mlp_accuracy_plot](https://user-images.githubusercontent.com/85929824/162430244-95342d2f-4fbd-434d-a343-9e7734e50198.png)

### Conclusion

The more layers we add the less the model is sensitive to parameters addition. In other words, if we have many layers, increasing their size has very few effects. On the other hand, with few layers, increasing their size has a strong effect on accuracy. We therefore find out that we already got a pretty good accuracy with only one dense layer of 512 entries. The optimal network size seems to be the one with 4 layers and the architecture 784x512x128x64x10 that sums up to 125898 parameters.

## Convolutional Neural Network (CNN)
