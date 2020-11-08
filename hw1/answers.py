r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""


#### Answer 1

A test dataset allow us to estimate how our model fit unseen data, and validate that we don't have a over-fitting problem.<br>
the test dataset follows the same probability distribution as the training dataset, so in an idle world it dosent matter,<br>
but we prefer to use random selected dataset. usually the test dataset will be smaller then the training set.

test-set usage on machine-learning pipeline:<ol>
**training**: not involved<br>
**cross-validation**: not involved<br>
**performance evaluation**: only with the data-set<br>
**deciding whether to use one trained model or another**: only with the data-set
</ol>
"""

part1_q2 = r"""
#### Answer 2
Yes, we do need to split out part of the training set as a validation set. The goal of using a validation set is to tune the hyperparameters.<br>
In order to avoid overfitting is critical to ensure that the training set and the validation set are not overlap.<br>
In addition the validation and the training dataset should be different as well, 
otherwise the test set will affect the hyperparameters choice and then the test would not be objective.

"""

# ==============
# Part 2 answers

part2_q1 = r"""
#### Answer 1
larger values of k reduces effect of the noise on the classification, because the impact of a single point on the prediction<br>
goes down as the k increase. the problem with big k is that is more difficult to distinct the boundaries  of the classes.<br>
In addition very big k is vulnerable to uneven training data (where there are more samples of one class then the others).<br>
in the extreme case where k=N, the prediction always will be the class with the majority of samples.
"""

part2_q2 = r"""
#### Answer 2
1. Training on the entire train-set with various models will probably cause a over-fitting situation in all the models,<br>
and using k-fold CV can find over-fitting in the training process.
2. Training on the entire train-set with various models with the default hyperparameters without tuning them will cause<br>
the accuracy to drop compare to the use with k-fold CV, and the best hyperparameters selection.
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
#### Answer 1
The selection of  Δ>0  is arbitrary for the SVM loss  𝐿(𝑾) because The hyperparameters Δ and λ seem like two different hyperparameters,
but in fact they both control the same tradeoff: The tradeoff between the data loss and the regularization loss in the objective.
The key to understanding this is that the magnitude of the weights W has direct effect on the scores (and hence also their differences):
As we shrink all values inside W the score differences will become lower, and as we scale up the weights the score differences will all become higher.
Therefore, the exact value of the margin between the scores (e.g. Δ=1, or Δ=100) is in some sense meaningless because the
weights can shrink or stretch the differences arbitrarily. Hence, the only real tradeoff is how large we allow the weights
to grow (through the regularization strength λ).
"""
"""Reference from: https://cs231n.github.io/linear-classify/"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
### Question 3
Based on the graph of the training set loss, would you say that the learning rate you chose is:
<ul>
 <li>Too low </li>
 <li><mark>Good</mark> </li>
 <li>Too High</ul> </li>
Explain your answer by describing what the loss graph would look like in the other two cases when training for the same number of epochs.<br>

**Too low**: the graph<br>
**Too High**: the graph<br>

Based on the graph of the training and test set accuracy, would you say that the model is:
<ul>
<li><mark>Slightly overfitted to the training set</mark></li>
<li>Highly overfitted to the training set</li>
<li>Slightly underfitted to the training set</li>
<li>Highly underfitted to the training set</ul> </li>
and why?

The validation curve is suspiciously close to the train curve.
"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
#### Answer 1
the ideal pattern to see in a residual plot is all the points exactly on the line.<br>
the fitness of the trained model is pretty good.<br>
the plot for the top-5 features is not that good.
"""

part4_q2 = r"""
1. Is this still a linear regression model? Yes! the linearity term is not refer to the linearity of the independent variables 
witch are being regressed against the dependent output.
2. Can we fit any non-linear function of the original features with this approach? No, like $e^{ax}$, $log(bx)$ where a and
b are weights.
3. The effect of adding non-linear features to our data is the same as adding more features to our data. It gives our model
more dimensions to find the right hyperplane.
</ul>
"""

part4_q3 = r"""
1. The effect between jumps of consecutive numbers is negligible, however a change in jumps of 10 times can be seen.
2.
"""

# ==============
