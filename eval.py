import os
import numpy as np
import pandas as pd
from keras.models import load_model

print('Getting models.')

start_dir = os.getcwd()
model_dir = start_dir + '/models/' 

model_names = os.listdir(model_dir)
if len(model_names) == 0:
    print("No models! Train!")
    exit

print('-' * 30)

print('Loading and preprocessing training and validation data...')

path = 'local_data/'

# We get the mean and std from the training data.
X = np.load(path + 'parts_train.npy')
y = np.load(path + 'parts_train_labels.npy')

X = X.astype('float32')
mean = np.mean(X)  # mean for data centering
std = np.std(X)  # std for data normalization

X = np.load(path + 'parts_validation.npy').astype('float32')
y = np.load(path + 'parts_validation_labels.npy').astype('float32')

X -= mean
X /= std

y /= 255.  # scale masks to [0, 1]

print('-' * 30)

print("Evaluating models in the models/ dir.")
eval_res = pd.DataFrame(columns=['loss', 'acc', 'param_count'])
# Let's evaluate the models
for model_name in model_names:
    print("Evaluating model " + model_name)
    try:
        model_file = model_dir + model_name + '/model.h5'
        model = load_model(model_file)
        res = model.evaluate(X, y)
        res.append(model.count_params())
        eval_res.loc[model_name] = res
    except OSError as e:
        print("No model file.")
    
    print('-' * 30)

eval_res.to_csv("eval_res.csv")

print("Evaluating Done!")
