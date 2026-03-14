Homework 1: Tiny up the notebook to be py scrips 

Ans: HW1_makemore_p2.py

Homework 2: Explain why the generated names are better than the previous lecture.

Ans: In previous lecture, the next character was predicted by the previous character, where the count is logits.exp() (or, prob = logits.exp() / sum(logits.exp())). In this lecture, the next character is predicted by 3 charcters and each character is extended to 2 (or more) dimensions. Also, a hidden layer with 100 (or more) neurons is added instead of bare w_i*x_i. Therefore, this complexity enhances the accuracy of prediction. 
