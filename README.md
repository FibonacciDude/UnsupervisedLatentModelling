
## Unsupervised Modelling of Latent Cognitive Characteristics

### Approach

I created a sequence model (Gated Recurrent Unit, GRU) to predict the next gaze given
all previous ones – auto-regression. After the model was trained, I took the hidden vector (64
dimensional) inside the GRU at each frame as the new data.
I then trained a small neural network model to predict which task the user was performing in the sequence
given only these hidden vectors. The network was given 32 total examples to generalize for all tasks and all
people.

The model performed with ~87% accuracy in predicting from the 4 tasks only using the hidden vectors. The
task of next-gaze prediction highly correlated with which task the user was performing. Furthermore, the
trained model’s hidden vector was high-fidelity as it allowed generalization from the network using only 32
examples.

I wished to explore which latent factors contribute to the behavior and motion of a person’s eye gaze. After
all, the eyes are the windows to the soul. So a model that predicts eye gaze could possibly predict certain
characteristics of a person’s thoughts/intentions.

### Data
The project uses a dataset of x,y coordinates of people’s eye gazes when 4 different computer tasks are being
performed. The coordinates are compiled through time until the task ends. The dataset is called “GazeBase”
https://www.nature.com/articles/s41597-021-00959-y.

To download data, I fetch from their website through the
https://github.com/FibonacciDude/UnsupervisedLatentModelling repository (different from training repo)
data/get_gazebase.sh and unzip by data/unzip_gazebase.sh.

Data is then converted to pytorch-compatible and extraneous data cleaned through running data_index.py.
When fed into the model, the sequence data is properly batched.

#### model.py file
The GRU model object is created using the config.json file and the task prediction model as well. The task
prediction is a simple feedforward network.

#### train.py file

Runs an automatic cross-validation of the GRU model and the task predictor model using the configurations
specified in config.json. Data is batched and the model is trained using this data for prediction of next x,y
gaze vectors using MSE loss (if GRU), or on task classification using the GRU hidden vectors (if predictor
model).
