## Activation Functions:

Activation function of a node defines the output of that node given an input or set of inputs.


<br>

**1. Sigmoid Function**

![sig](https://github.com/siddarthjha/ML/blob/master/Images/sig.png)

![fun](https://github.com/siddarthjha/ML/blob/master/Images/sig1.png)

Sigmoid functions are one of the most widely used activation functions today


<br>

**2. Tanh Function**

![tanh](https://github.com/siddarthjha/ML/blob/master/Images/tanh.png)

![fun](https://github.com/siddarthjha/ML/blob/master/Images/tanh1.png)

This looks very similar to sigmoid. In fact, it is a scaled sigmoid function!

Tanh is also a very popular and widely used activation function.


<br>

**3. Rectified linear Unit(RelU)**

![relu](https://github.com/siddarthjha/ML/blob/master/Images/relu'.jpg)

Later, comes the ReLu function,


A(x) = max(0,x)


The ReLu function is as shown above. It gives an output x if x is positive and 0 otherwise.


<br>

**4. Leaky Rectified Linear Unit**

**Relu vs Leaky Relu**

![lrelu](https://github.com/siddarthjha/ML/blob/master/Images/leay.jpeg)

* It is an attempt to solve the dying ReLU problem
* The leak helps to increase the range of the ReLU function. Usually, the value of a is 0.01 or so.
* When **a is not 0.01** then it is called **Randomized ReLU.**
* Therefore the range of the Leaky ReLU is (-infinity to infinity).
* Both Leaky and Randomized ReLU functions are monotonic in nature. Also, their derivatives also monotonic in nature.
