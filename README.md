# pytkml
Write tests for machine learning models

(This is a proof of concept that is still in progress. It is not fit for any purpose and everything about it is likely to change)

The idea is to be able to write a set of tests you want your model to pass. The focus is on testing the workings of the model, rather than the output or performance. 

The figure below shows how the tests work.

![](test_flow_m.png)

A ModelTester combines a model (currently PyTorch classifiers) with test data and tests. For each test, batches of data are processed on parallel pathways considering the sample and label. After processing, these are compared, and a `True` comparison corresponds to a passed test.

There is currently one nontrivial test. Trivial tests like accuracy are also possible. The `influence_transform` ranks the samples in a supplied training set according to the cosine similarity between each sample's gradient and that of the test samples. This has been proposed as a way to identify training instances that are helpful or harmful to classifying a point [1]. Intuitively, if a test sample has good support in the training data, you would expect the most similar training points to be of the same class. Likewise, you might expect harmful training points to look similar but be of a different class.

The above idea can be rolled into a test, where we compare the actual classes with the influential classes for a set of data, to provide evidence that the model is working how we expect. The current implementation performs these tests, and saves the results in a log which can be recalled and visualized. The plan is to add more such tests. 

[Colab Example (MNIST)](https://colab.research.google.com/drive/1auylRHGuWIR9cZiP92eCLj5QWtj6EjQr?usp=sharing)

[Second Colab Example (MVTEC)](https://colab.research.google.com/drive/1fb3mrLRNUspaWdKzkniGY4W1jUFs7dZP?usp=sharing)

[1] Hanawa et al., https://arxiv.org/abs/2006.04528

[Todo] There are lots of other relevant references.

Contact: andrew@willows.ai

