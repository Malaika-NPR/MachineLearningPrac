from torch import no_grad
from torch.utils.data import DataLoader


"""
Functions you should use.
Please avoid importing any other functions or modules.
Your code will not pass if the gradescope autograder detects any changed imports
"""
from torch import optim, tensor
from losses import regression_loss, digitclassifier_loss, languageid_loss, digitconvolution_Loss
from torch import movedim


"""
##################
### QUESTION 1 ###
##################
"""


def train_perceptron(model, dataset):
    """
    Train the perceptron until convergence.
    You can iterate through DataLoader in order to 
    retrieve all the batches you need to train on.

    Each sample in the dataloader is in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.
    """
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    "*** YOUR CODE HERE ***"
    # Continuous looping through the dataloader 
    Train = True
    while Train:
        correctly_classified = True
        for batches in dataloader: 
            x = batches['x']
            label = batches['label'].item()
            get_prediction = model.get_prediction(x)
            # Making updates on examples that are misclassified 
            if get_prediction != label:
                direction = x[0]
                magnitude = label 
                with no_grad():
                    model.w.data += direction * magnitude  
                correctly_classified = False
            # Has 100% training accuracy - terminate training 
        if correctly_classified:
            break


def train_regression(model, dataset):
    """
    Trains the model.

    In order to create batches, create a DataLoader object and pass in `dataset` as well as your required 
    batch size. You can look at PerceptronModel as a guideline for how you should implement the DataLoader

    Each sample in the dataloader object will be in the form {'x': features, 'label': label} where label
    is the item we need to predict based off of its features.

    Inputs:
        model: Pytorch model to use
        dataset: a PyTorch dataset object containing data to be trained on
        
    """
    "*** YOUR CODE HERE ***"


def train_digitclassifier(model, dataset):
    """
    Trains the model.
    """
    model.train()
    """ YOUR CODE HERE """
    dataLoader = DataLoader(dataset, batch_size = 64, shuffle = True)
    learningRate = 0.01
    optimizer = optim.Adam(model.parameters(), lr = learningRate)
    # Test training data and iterate through all of the batches to classify 
    for epoch in range(100):
        for batch in dataLoader:
            x = batch['x']    
            y = batch['label']   
            optimizer.zero_grad()
            prediction = model(x)
            loss = digitclassifier_loss(prediction, y)
            loss.backward()
            optimizer.step()
        val_accuracy = dataset.get_validation_accuracy()
        if val_accuracy >= 0.975:
            break
        #Problem Sources
        #https://www.youtube.com/watch?v=oUuf2O37VAc
        #https://www.geeksforgeeks.org/machine-learning/handwritten-digit-recognition-using-neural-network/
        #https://medium.com/@koushikkushal95/mnist-hand-written-digit-classification-using-neural-network-from-scratch-54da85712a06
        #https://docs.pytorch.org/tutorials/recipes/recipes/zeroing_out_gradients.html




def train_languageid(model, dataset):
    """
    Trains the model.

    Note that when you iterate through dataloader, each batch will returned as its own vector in the form
    (batch_size x length of word x self.num_chars). However, in order to run multiple samples at the same time,
    get_loss() and run() expect each batch to be in the form (length of word x batch_size x self.num_chars), meaning
    that you need to switch the first two dimensions of every batch. This can be done with the movedim() function 
    as follows:

    movedim(input_vector, initial_dimension_position, final_dimension_position)

    For more information, look at the pytorch documentation of torch.movedim()
    """
    model.train()
    "*** YOUR CODE HERE ***"



def Train_DigitConvolution(model, dataset):
    """
    Trains the model.
    """
    """ YOUR CODE HERE """
