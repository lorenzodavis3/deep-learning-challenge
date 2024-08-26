# deep-learning-challenge


Alphabet Soup Deep Learning Model Report

    Overview
        The goal was to create a machine learning model capable of predicting the success of applications for funding from Alphabet Soup. Using a dataset of historical records of funded organizations, I built a binary classification model that would predict the success of future funding based on key features provided in the dataset. The aim was to use deep learning, specifically a neural network, to build this predictive model and optimize its performance to achieve an accuracy rate of 75% or higher.

    Results
        Data Preprocessing
            Target Variable:
                The target variable for this model is IS_SUCCESSFUL, which indicates whether the funded organization was successful.
        Feature Variables:
            The features used in the model include:
            - APPLICATION_TYPE
            - AFFILIATION
            - CLASSIFICATION
            - USE_CASE
            - ORGANIZATION
            - STATUS
            - INCOME_AMT
            - SPECIAL_CONSIDERATIONS
            - ASK_AMT
        Removed Variables:
            The EIN and NAME columns were removed from the dataset because they are identification numbers that do not contribute meaningful information for predicting success.
    
    Compiling, Training, and Evaluating the Model
        Neural Network Model Structure:
            The final optimized model was built with the following structure:
                - First hidden layer: 256 neurons, relu activation
                - Second hidden layer: 128 neurons, relu activation
                - Third hidden layer: 64 neurons, relu activation
                - Output layer: 1 neuron, sigmoid activation
                - Dropout layers: Dropout layers with a rate of 0.3 were added after the first and second hidden layers to prevent overfitting.
        Performance Results:
            - Original Model: The original model achieved an accuracy of 72.52% with a loss of 0.5655.
            - First Optimization (Increased neurons and added layers): My first attempt increased the number of neurons and layers but resulted in an accuracy of 72.73% and a loss of 0.5999, which was worse than the original.
            - Second Optimization (Changed activation functions and added dropout layers): The second attempt, which used the tanh activation function and included dropout layers, yielded an accuracy of 72.32% and a loss of 0.5618, which was worse than the original and the first optimization.
            - Final Optimized Model: The final optimized model reverted back to relu activations, increased neurons in the hidden layers, increased the dropout rate to 0.3, and added early stopping. This model achieved the best performance with an accuracy of 73.09% and a loss of 0.5606.
        Optimization Steps:
            - Increased the number of neurons in the first and second hidden layers to 256 and 128, respectively.
            - Introduced dropout layers with a rate of 0.3 to prevent overfitting.
            - Implemented early stopping to halt training once the model's loss no longer improved.
            - Reverted back to the relu activation function after experimenting with tanh, which showed poorer performance.

    Summary
        The final deep learning model achieved an accuracy of 73.09%, which is an improvement over the initial model but did not meet the target performance of 75%. While progress was made by optimizing the network structure, adding dropout layers, and implementing early stopping, the model still fell short of the 75% threshold.
        To achieve better results, it may be beneficial to try alternative machine learning models such as Random Forests or Support Vector Machines (SVM).

Sources:
- Use of dropout regulariation: https://machinelearningmastery.com/dropout-regularization-deep-learning-models-keras/
- Use of EarlyStopping: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
- Save in HDF5 format: https://machinelearningmastery.com/save-load-keras-deep-learning-models/