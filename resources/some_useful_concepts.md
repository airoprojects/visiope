
# Metrics

1. Pixel Accuracy is bad for our case since the dataset is unbalanced, most of pixels the images are background pixel so even a all black prediction would give an high score for this metric.



# Encoding

1. The cross-entropy loss function will compute the element-wise softmax activation on the prediction tensor along the class dimension to convert it into a probability distribution. It will then compare this predicted probability distribution with the ground truth labels using the negative log-likelihood loss calculation. Therefore, even though the dimensions of the prediction and label tensors differ, the cross-entropy loss function in PyTorch will appropriately handle the discrepancy and calculate the loss on a
per-pixel basis.