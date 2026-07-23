A simple practice of Physics-Informed Neutral Networks (PINNs). This project explores several basic applications of Physics-Informed Neural Networks (PINNs) under Gauss’s law. For comparison, a traditional numerical method—the Jacobi method (Gauss-Jacobi.py)—is also implemented. Jupyter notebook is also provided as a reference.

Parctice 1: Laplace-PINNs
- No net charge is enclosed, so \rho(x, y) = 0. Gauss’s law reduces to the Laplace equation. 
- The top and bottom boundaries are fixed at 1 V, while the left and right boundaries are 0 V.
- The model is trained for 30,000 steps, achieving a final loss of 0.0467 and an error of 0.0834.
- The results show good agreement between the Jacobi method and PINNs, except at the four corners. PINNs struggle with corner values because the boundary conditions are manually imposed.  
- The Jacobi method finishes in under 1 minute, whereas PINNs require significantly more time. This indicates that PINNs are not efficient for such a simple scenario.

Parctice 2: Gauss's law - PINNs
- The geometry is identical to Practice 1.
- Two net charges are enclosed, with charge-to-\epsilon_0 ratios of +2 and –2. The charge density is defined as \rho = q / (\pi r_s^2), where r_s is charge radius. This linear approximation differs from the real physical distribution but is sufficient for practice.
- The top and bottom boundaries are set to 0.8 V, while the left and right boundaries remain 0 V.
- Batch sizes are increased, and the maximum number of training steps is raised to 50,000. The final error is 0.166, which is relatively high.
- The Jacobi method completes in a few minutes, whereas PINNs require about 4 hours.

Parctice 3: Gauss's law - data imported PINNs
- All settings are the same as in Practice 2.
- Additional data (batch size = 1000) are imported to assist training.
- The maximum number of steps is set to 30,000, resulting in a final error of 0.0753, roughly half of that in Practice 2.
- The Jacobi method again finishes within a few minutes, while PINNs take around 3 hours.