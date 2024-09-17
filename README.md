# Improved-Training-GTZAN-Dataset
Various files that show a novel method of neural network training, repeatedly randomizing the weights and biases and directly calculating accuracy without training, then training the best of these unoptimized configurations, showing greater performance than the Adam optimizer on the GTZAN dataset.

Note that the file names and code structure are not cleaned up for presentation but are functional. 

This was done as a final project for machine learning under professor daniel leeds as a group project, but the nerual network section was written entirely by me. Please look at the PDF section on neural networks for more detail and analysis.

To run:

Have the .py file to be run in the same folder as features_30_sec.csv

Have python installed from the normal python website. My version is 3.11.2 but this shouldn't matter

Install the needed packages if not already installed
pip install numpy
pip install pandas
pip install scipy            
(scipy needed for sklearn to work)
pip install scikit-learn

for keras additionally it is needed to install tensorflow
pip install tensorflow

Run
python RandomizationVersion1Timed.py
or whatever filename desired in the same format.

To interpret the results:
Initally it will print out the data of the csv file to confirm it. 
If there is randomization, the python file will print out Test Accuracy: XXX while not giving out an epoch. This is because there are no epochs in randomization.

To change randomization time, just go to 
#Randomization
current_best = float('inf')
for i in range(10000):

and change the counter to however many numbers desired.

The expected output is that Epochs begin to be counted and given a test accuracy and loss. 
It is usually the case that the first 10000 epochs will seem to show no accuracy improvement and then spontaneously improve.
It is also usually the case that after a peak of around 73 to 75 test accruacy will go down, and seems to have a tendency to do so around .68 or higher.
This is for the 80 20 split, the 60 40 converges around 71 or 72. 
Occasionally Keras or the ramdonization method will show .78 but this is rare and unreliable. It suggests however that if this techinque were used in tens or hundreds of iterations, it may show better performance.
