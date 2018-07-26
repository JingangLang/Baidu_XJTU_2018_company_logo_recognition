import os
import shutil

import pandas as pd

validation_split = 0.0

for root, dirs, files in os.walk('../data', topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
os.mkdir('../data/train')
os.mkdir('../data/validation')


train = pd.read_csv('../datasets/train.txt', sep=' ', header=None, names=['filename', 'label'])
train = train.sample(frac=1).reset_index(drop=True) # shuffle data

split_index = int(train.shape[0] * (1 - validation_split))

validation = train.iloc[split_index:]
train = train.iloc[:split_index]

print(train.shape)
print(validation.shape)

classes = train['label'].drop_duplicates().tolist()
for clas in classes:
    if not os.path.exists('../data/train/' + str(clas)):
        os.mkdir('../data/train/' + str(clas))
        
    file_list = train[train['label']==clas]['filename'].tolist()
    for file_name in file_list:
        shutil.copy('../datasets/train/' + file_name, '../data/train/' + str(clas) + '/' + file_name)

classes = validation['label'].drop_duplicates().tolist()
for clas in classes:
    if not os.path.exists('../data/validation/' + str(clas)):
        os.mkdir('../data/validation/' + str(clas))
        
    file_list = validation[validation['label']==clas]['filename'].tolist()
    for file_name in file_list:
        shutil.copy( '../datasets/train/' + file_name, '../data/validation/' + str(clas) + '/' + file_name)

for _, dirs, _ in os.walk('../extra'):
	for claz in dirs:
		for _, _, files in os.walk('../extra/' + claz):
			for file in files:
				shutil.copy('../extra/' + claz + '/' + file, '../data/train/' + claz + '/' + file)
