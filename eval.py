import os
import argparse
import sys
import numpy as np
import pandas as pd
from keras.models import load_model
from BilinearUpSampling import BilinearUpSampling2D

parser = argparse.ArgumentParser(
    description='Evaluate the model')
parser.add_argument('file_name', metavar='file_name', type=str,
                    help='name for the csv file')
parser.add_argument('-m', '--model', metavar='model_name', type=str,
                    help='model name that are evaluated', default="")
parser.add_argument('-t', '--test', action='store_true', 
                    help='Set to use test data set. Default is the validation set.')

args = parser.parse_args()
file_name = args.file_name
pattern = args.model

print('Getting models.')
if (len(sys.argv) == 2):
    pattern = sys.argv[1]

start_dir = os.getcwd()
model_dir = start_dir + '/models/' 

model_names = os.listdir(model_dir)
if len(model_names) == 0:
    print("No models! Train!")
    exit

print('-' * 30)

print('Loading training to calculate mean and variance...')

path = 'local_data/'

# We get the mean and std from the training data.
X = np.load(path + 'parts_train.npy')
y = np.load(path + 'parts_train_labels.npy')

X = X.astype('float32')
mean = np.mean(X)  # mean for data centering
std = np.std(X)  # std for data normalization

X = np.array([])
y = np.array([])
print('-' * 30)

if args.test:
    print('Loading test data...')
    X = np.load(path + 'parts_test.npy').astype('float32')
    y = np.load(path + 'parts_test_labels.npy').astype('float32')
else:
    print('Loading validation data...')
    X = np.load(path + 'parts_validation.npy').astype('float32')
    y = np.load(path + 'parts_validation_labels.npy').astype('float32')

X -= mean
X /= std

y /= 255.  # scale masks to [0, 1]

print('-' * 30)

print("Evaluating models in the models/ dir.")
names = ['Model', 'Loss', 'Accuracy', 'Parameters']
eval_res = pd.DataFrame(columns=names)
# Let's evaluate the models
for model_name in model_names:
    if (pattern in model_name):
        print("Evaluating model " + model_name)
        try:
            model_file = model_dir + model_name + '/model.h5'
            model = load_model(model_file, 
                    custom_objects={'BilinearUpSampling2D': BilinearUpSampling2D})
            res = model.evaluate(X, y)
            res.append(model.count_params())
            res.insert(0, model_name)
            df_res = pd.DataFrame([res], columns=names)
            eval_res = eval_res.append(df_res, ignore_index=True)
        except OSError as e:
            print("No model file.")
        
        print('-' * 30)

eval_res['Loss'] = eval_res['Loss'].round(3)
eval_res['Accuracy'] = (eval_res['Accuracy']*100).round(2)
# eval_res['Parameters'] = eval_res['Parameters'].round(0)
eval_res = eval_res.sort_values(by=['Model'])

eval_res.to_csv(file_name+".csv", index=False)

print("Evaluating Done!")
