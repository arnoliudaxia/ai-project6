import matplotlib.pyplot as plt
import numpy as np

import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w,x)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x))>=0:
            return 1
        return -1

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        loopFlage=True
        while loopFlage:
            loopFlage = False
            for x, y in dataset.iterate_once(1):
                result = self.get_prediction(x)
                if result != nn.as_scalar(y):
                    self.w.update(nn.Constant(x.data), nn.as_scalar(y) )
                    loopFlage = True

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningRate=1e-1
        self.batchSize=10
        self.layer1w=nn.Parameter(1, 256)
        self.layer1b=nn.Parameter(1, 256)
        self.layer2w=nn.Parameter(256, 32)
        self.layer2b=nn.Parameter(1, 32)
        self.layer3w=nn.Parameter(32, 1)
        self.layer3b=nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        layer1output=nn.AddBias(nn.Linear(x, self.layer1w), self.layer1b)
        layer2output=nn.AddBias(nn.Linear(nn.ReLU(layer1output), self.layer2w), self.layer2b)
        layer3output=nn.AddBias(nn.Linear(nn.ReLU(layer2output), self.layer3w), self.layer3b)
        return layer3output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictionResult=self.run(x)
        loss = nn.SquareLoss(predictionResult, y)
        return loss



    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        stopFlag=False
        params=[self.layer1w,self.layer1b,self.layer2w,self.layer2b,self.layer3w,self.layer3b]
        while not stopFlag:
            stopFlag = True
            for x, y in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(x, y)
                lossNum=nn.as_scalar(loss)
                # print(f"loss: {lossNum}")
                grads = nn.gradients(loss, params)
                for i in range(len(params)):
                    params[i].update(grads[i], -self.learningRate)
                if lossNum>0.015:
                    stopFlag=False

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.learningRate=5e-1
        self.batchSize=100
        self.layer1w=nn.Parameter(784, 300)
        self.layer1b=nn.Parameter(1, 300)
        self.layer2w=nn.Parameter(300, 128)
        self.layer2b=nn.Parameter(1, 128)
        self.layer3w=nn.Parameter(128, 64)
        self.layer3b=nn.Parameter(1, 64)
        self.layer4w=nn.Parameter(64, 10)
        self.layer4b=nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        layer1output=nn.AddBias(nn.Linear(x, self.layer1w), self.layer1b)
        layer2output=nn.AddBias(nn.Linear(nn.ReLU(layer1output), self.layer2w), self.layer2b)
        layer3output=nn.AddBias(nn.Linear(nn.ReLU(layer2output), self.layer3w), self.layer3b)
        layer4output=nn.AddBias(nn.Linear(nn.ReLU(layer3output), self.layer4w), self.layer4b)
        return layer4output

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        predictionResult=self.run(x)
        loss = nn.SoftmaxLoss(predictionResult, y)
        return loss

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"


        stopFlag=False
        params=[self.layer1w,self.layer1b,self.layer2w,self.layer2b,self.layer3w,self.layer3b,self.layer4w,self.layer4b]
        # print("加载预训练权重")
        # pretainedFile="best.npz"
        # pretainedWeights=np.load(pretainedFile)
        # # self.learningRate *= 0.6**3
        # for i in range(len(params)):
        #     params[i].data=pretainedWeights[f"arr_{i}"]
        iterationCounter=0
        # print("开始训练")
        # stopFlag=True
        while not stopFlag:
            iterationCounter +=1
            stopFlag = False
            lossHistory=[]
            for x, y in dataset.iterate_once(self.batchSize):
                loss = self.get_loss(x, y)
                lossNum=nn.as_scalar(loss)
                # print(f"loss: {lossNum}")
                lossHistory.append(lossNum)
                grads = nn.gradients(loss, params)
                for i in range(len(params)):
                    params[i].update(grads[i], -self.learningRate)
            plt.plot(lossHistory)
            plt.show()
            ValAcc=dataset.get_validation_accuracy()
            print(f"Validation Accuarcy is {ValAcc}")
            self.learningRate*=0.9
            if ValAcc>0.975:
                stopFlag = True
            # print("保存权重中")
            # np.savez_compressed(f"weights/dig-{iterationCounter}",params[0].data,params[1].data,
            #          params[2].data,params[3].data,
            #          params[4].data,params[5].data,
            #          params[6].data,params[7].data)
            # print("保存完成")


class LanguageIDModel(object):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.initialW = nn.Parameter(self.num_chars, 1024)
        self.initialB = nn.Parameter(1, 1024)
        self.W = nn.Parameter(self.num_chars, 1024)
        self.HW = nn.Parameter(1024, 1024)
        self.B = nn.Parameter(1, 1024)
        self.outputLayW = nn.Parameter(1024, len(self.languages))
        self.outputLayB = nn.Parameter(1, len(self.languages))
        # self.params = [self.initialW, self.initialB, self.W,self.HW, self.B,
        #                self.outputLayW, self.outputLayB]

    def run(self, xs):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here `xs` will be a list of length L. Each element of `xs` will be a
        node with shape (batch_size x self.num_chars), where every row in the
        array is a one-hot vector encoding of a character. For example, if we
        have a batch of 8 three-letter words where the last word is "cat", then
        xs[1] will be a node that contains a 1 at position (7, 0). Here the
        index 7 reflects the fact that "cat" is the last word in the batch, and
        the index 0 reflects the fact that the letter "a" is the inital (0th)
        letter of our combined alphabet for this task.

        Your model should use a Recurrent Neural Network to summarize the list
        `xs` into a single node of shape (batch_size x hidden_size), for your
        choice of hidden_size. It should then calculate a node of shape
        (batch_size x 5) containing scores, where higher scores correspond to
        greater probability of the word originating from a particular language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
        Returns:
            A node with shape (batch_size x 5) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        #The first Word
        hi = nn.ReLU(nn.AddBias(nn.Linear(xs[0], self.initialW), self.initialB))
        # the rest words
        for char in xs[1:]:
            hi = nn.ReLU(nn.AddBias(nn.Add(nn.Linear(char, self.W),nn.Linear(hi, self.HW)), self.B))
        return nn.AddBias(nn.Linear(hi, self.outputLayW), self.outputLayB)

    def get_loss(self, xs, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 5). Each row is a one-hot vector encoding the correct
        language.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a node with shape (batch_size x self.num_chars)
            y: a node with shape (batch_size x 5)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(xs), y)

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        self.learningRate = .3
        stopFlag=False
        params=[self.initialW, self.initialB, self.W,self.HW, self.B,
                       self.outputLayW, self.outputLayB]
        iterationCounter=0
        print("开始训练")
        batchSize=100
        while not stopFlag:
            iterationCounter += 1
            stopFlag = False
            lossHistory = []
            for x, y in dataset.iterate_once(batchSize):
                loss = self.get_loss(x, y)
                lossNum = nn.as_scalar(loss)
                # print(f"loss: {lossNum}")
                lossHistory.append(lossNum)
                grads = nn.gradients(loss, params)
                for i in range(len(params)):
                    params[i].update(grads[i], -self.learningRate)
            plt.plot(lossHistory)
            plt.show()
            ValAcc = dataset.get_validation_accuracy()
            print(f"Validation Accuarcy is {ValAcc}")
            self.learningRate *= 0.8
            if ValAcc > 0.84:
                stopFlag = True

            # print("保存权重中")
            # np.savez_compressed(f"weights/it-{iterationCounter}",params[3].data)
            # # np.savez(f"weights/{iterationCounter}",params[0].data,params[1].data,
            # #          params[2].data,params[3].data,
            # #          params[4].data,params[5].data,
            # #          params[6].data)
            # print("保存完成")
