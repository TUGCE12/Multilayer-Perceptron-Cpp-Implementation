# Multilayer-Perceptron-Cpp-Implementation
 MLP C++ project
 
 For this project:
  * StatQuest with Josh Starmer / Neural Networks / Deep Learning /-
  * -> https://www.youtube.com/watch?v=zxagGtF9MeU&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=2
  * The first 8 videos in the playlist were used.
  * There is no code in the video series (at least in the first 8)
  * It is explained theoretically only.
  * This project is a C++ implementation of the theoretically explained MLP problem and BackPropagation example.
  * Example Problem: A pharmaceutical company has tested 3 groups for the drug it produces.
  * 1. Group: Low dose drug was applied. Viruses are not dead. (Unsuccessful)
  * 2. Group: Normal dose of drug was applied. The viruses are dead. (Successful)
  * 3rd Group: High dose drug was applied. Viruses are not dead. (Unsuccessful)
  * For simplicity, drug doses are taken as low = 0 and high = 1.
  * Success =1 and Fail = 0 .
  * Dataset = [[0.0,0.0],[0.5,1.0],[1.0,0.0]] .
  * Neural Network will be used to find a model that will fit this.

  |&emsp;&emsp;&emsp;\*
  
  |
  
  |\*__________\*__ The data distribution is like this.
  * To draw a graph suitable for this data, I need 2 graphs in the form of y = ax + b.
  * In video was used 2 nodes in Hidden Layer.
    
  * *****************************************
Number of input nodes = 1
Number of nodes per hidden layer = 2
Number of output nodes = 1
Number of hidden layers = 1
The learning rate = 0.015
Loss Fuction = Sum of squares Residual
Activation function = SoftPlus = f(x) = log(1+e^x)



Redidual = Observed - Predicted


 Neural Network:
n= Number of nodes per hidden layer
i = Number of data

|Input[i]|---(x W1[n] )---( + B1[n] )---------|Hidden Layer (n)|---( x W2[n] )------(Sum) + B---|Output|


 (-->SumFunction +B for a x)
 Neural Network Function is = E(i=0->i=2) [B + W2i * softPlus( x * W1i + B1i )]

 E = Sum

