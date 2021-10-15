# FirstClass_LSPI
This is my first Python programm which is created for the implementation of the Least Squares Policy Iteration (LSPI) reinforcement learning algorithm. 
For more information on the algorithm please refer to the paper

“Least-Squares Policy Iteration.”
Lagoudakis, Michail G., and Ronald Parr.
Journal of Machine Learning Research 4, 2003.
https://www.cs.duke.edu/research/AI/LSPI/jmlr03.pdf

You can also visit their website where more information and a Matlab version is provided.
http://www.cs.duke.edu/research/AI/LSPI/

This programm also refers to Devin's project 
https://pythonhosted.org/lspi-python/index.html

However, some mistakes have been found therein based on the knowlege that I have got before. So, I have rewritten this project accroding to my understanding at that time. The project contains the following four functions:

1) main.py  this function is the main boby of the LSPI algorithm and provides the visulization of the results. This function also calls the following four subfunctions. 

2) lstdq.py, which is written to evalue the Q-value of a given policy by LSTDQ algorithm.

3) Data_sample.py, which is the function used to collect the sample data. This set of sample data is fed to LSTDQ for obtaining the Q_value of differet policy. This is the so-called experience replay.

4) basisfunc.py, which is the achievement of basis function. Here, we just give the type of polinomal functions.

If you just want to see whether the LSPI algorithm can work, it is better to run the main.py directly and it will give you the optimal policy for the chain problem.
