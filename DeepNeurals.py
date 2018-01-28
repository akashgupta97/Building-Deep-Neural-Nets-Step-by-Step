
import numpy as np
import h5py
import matplotlib.pyplot as plt
from testCases_v2 import *
from dnn_utils_v2 import sigmoid, sigmoid_backward, relu, relu_backward

get_ipython().magic('matplotlib inline')
plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

np.random.seed(1)


# **Expected output**:
#
# <table style="width:80%">
#   <tr>
#     <td> **W1** </td>
#     <td> [[ 0.01624345 -0.00611756]
#  [-0.00528172 -0.01072969]] </td>
#   </tr>
#
#   <tr>
#     <td> **b1**</td>
#     <td>[[ 0.]
#  [ 0.]]</td>
#   </tr>
#
#   <tr>
#     <td>**W2**</td>
#     <td> [[ 0.00865408 -0.02301539]]</td>
#   </tr>
#
#   <tr>
#     <td> **b2** </td>
#     <td> [[ 0.]] </td>
#   </tr>
#
# </table>

# ### 3.2 - L-layer Neural Network
#
# The initialization for a deeper L-layer neural network is more complicated because there are many more weight matrices and bias vectors. When completing the `initialize_parameters_deep`, you should make sure that your dimensions match between each layer. Recall that $n^{[l]}$ is the number of units in layer $l$. Thus for example if the size of our input $X$ is $(12288, 209)$ (with $m=209$ examples) then:
#
# <table style="width:100%">
#
#
#     <tr>
#         <td>  </td>
#         <td> **Shape of W** </td>
#         <td> **Shape of b**  </td>
#         <td> **Activation** </td>
#         <td> **Shape of Activation** </td>
#     <tr>
#
#     <tr>
#         <td> **Layer 1** </td>
#         <td> $(n^{[1]},12288)$ </td>
#         <td> $(n^{[1]},1)$ </td>
#         <td> $Z^{[1]} = W^{[1]}  X + b^{[1]} $ </td>
#
#         <td> $(n^{[1]},209)$ </td>
#     <tr>
#
#     <tr>
#         <td> **Layer 2** </td>
#         <td> $(n^{[2]}, n^{[1]})$  </td>
#         <td> $(n^{[2]},1)$ </td>
#         <td>$Z^{[2]} = W^{[2]} A^{[1]} + b^{[2]}$ </td>
#         <td> $(n^{[2]}, 209)$ </td>
#     <tr>
#
#        <tr>
#         <td> $\vdots$ </td>
#         <td> $\vdots$  </td>
#         <td> $\vdots$  </td>
#         <td> $\vdots$</td>
#         <td> $\vdots$  </td>
#     <tr>
#
#    <tr>
#         <td> **Layer L-1** </td>
#         <td> $(n^{[L-1]}, n^{[L-2]})$ </td>
#         <td> $(n^{[L-1]}, 1)$  </td>
#         <td>$Z^{[L-1]} =  W^{[L-1]} A^{[L-2]} + b^{[L-1]}$ </td>
#         <td> $(n^{[L-1]}, 209)$ </td>
#     <tr>
#
#
#    <tr>
#         <td> **Layer L** </td>
#         <td> $(n^{[L]}, n^{[L-1]})$ </td>
#         <td> $(n^{[L]}, 1)$ </td>
#         <td> $Z^{[L]} =  W^{[L]} A^{[L-1]} + b^{[L]}$</td>
#         <td> $(n^{[L]}, 209)$  </td>
#     <tr>
#
# </table>
#
# Remember that when we compute $W X + b$ in python, it carries out broadcasting. For example, if:
#
# $$ W = \begin{bmatrix}
#     j  & k  & l\\
#     m  & n & o \\
#     p  & q & r
# \end{bmatrix}\;\;\; X = \begin{bmatrix}
#     a  & b  & c\\
#     d  & e & f \\
#     g  & h & i
# \end{bmatrix} \;\;\; b =\begin{bmatrix}
#     s  \\
#     t  \\
#     u
# \end{bmatrix}\tag{2}$$
#
# Then $WX + b$ will be:
#
# $$ WX + b = \begin{bmatrix}
#     (ja + kd + lg) + s  & (jb + ke + lh) + s  & (jc + kf + li)+ s\\
#     (ma + nd + og) + t & (mb + ne + oh) + t & (mc + nf + oi) + t\\
#     (pa + qd + rg) + u & (pb + qe + rh) + u & (pc + qf + ri)+ u
# \end{bmatrix}\tag{3}  $$

#
# **Instructions**:
# - The model's structure is *[LINEAR -> RELU] $ \times$ (L-1) -> LINEAR -> SIGMOID*. I.e., it has $L-1$ layers using a ReLU activation function followed by an output layer with a sigmoid activation function.
# - Use random initialization for the weight matrices. Use `np.random.rand(shape) * 0.01`.
# - Use zeros initialization for the biases. Use `np.zeros(shape)`.
# - We will store $n^{[l]}$, the number of units in different layers, in a variable `layer_dims`. For example, the `layer_dims` for the "Planar Data classification model" from last week would have been [2,4,1]: There were two inputs, one hidden layer with 4 hidden units, and an output layer with 1 output unit. Thus means `W1`'s shape was (4,2), `b1` was (4,1), `W2` was (1,4) and `b2` was (1,1). Now you will generalize this to $L$ layers!
# - Here is the implementation for $L=1$ (one layer neural network). It should inspire you to implement the general case (L-layer neural network).
# ```python
#     if L == 1:
#         parameters["W" + str(L)] = np.random.randn(layer_dims[1], layer_dims[0]) * 0.01
#         parameters["b" + str(L)] = np.zeros((layer_dims[1], 1))
# ```

# In[4]:

# GRADED FUNCTION: initialize_parameters_deep

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)  # number of layers in the network

    for l in range(1, L):
        ### START CODE HERE ### (≈ 2 lines of code)
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        ### END CODE HERE ###

        assert (parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l - 1]))
        assert (parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


# In[5]:

parameters = initialize_parameters_deep([5, 4, 3])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

