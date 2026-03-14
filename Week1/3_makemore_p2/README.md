Knowledge: 
1. Different from a single character, 3 characters (= block size) has been used to predict the next character, and each character is entended into 2 or more dimensional vectors. Therfore, the total number in input layer becomes (input_size, block size * dimension). Then, a hidden layer with 100 or more neurons has been used, where the operation is h = tanh(w^p_j * v^p_i + b_j), where j is the order of neutron and v^p_i vector spanned by the input x_i in the pth order direction (Summation convention is used). After the operator of output layer, h @ w2 + b2, the tensor shape becomes (unput size, 27); thus, the counts, probability, and its loss (dimension * block size -> a single value by sum of them).

2. The skill of splitting the input data has been learnt. First, shuffle the data into a random order; then, split the data into 80%/10%/10% distribution, of which 80%, 10% and 10% are for training, development and testing. As the name suggests, training set is for training the model, develpment set is for tuning the choices of parameters like learning rate and number of neurons, and testing set is for testing the availability of the model.

3. As using all input data for training at once is not efficient (slow), sampling the input data (32, 64, 128, ...) should be considered. For a proper training, the final loss of the sample-training model will not have too much differences from the actual loss.  

Homework:
1. Tiny up the notebook to be py scrips. Ans: HW1_makemore_p2.py

2. Explain why the generated names are better than the previous lecture. Ans: In previous lecture, the next character was predicted by the previous character, where the count is logits.exp() (or, prob = logits.exp() / sum(logits.exp())). In this lecture, the next character is predicted by 3 charcters and each character is extended to 2 (or more) dimensions. Also, a hidden layer with 100 (or more) neurons is added instead of bare w_i*x_i. Therefore, this complexity enhances the accuracy of prediction. 

References:
1. Neural Networks: Zero to Hero, Building makemore Part 2: MLP [https://www.youtube.com/watch?v=TCH_1BHY58I]
2. Y. Bengio, R. Ducharme, P. Vincent, C. Jauvin, A Neural Probabilistic Language Model, JMLR 3, 1137-1155 (2003).
