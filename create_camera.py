import sys
import ast
import numpy as np
import pickle


camera_file = sys.argv[1]
f = open(camera_file)
camera_dict = ast.literal_eval(f.read())
print(camera_dict.keys())
for i in camera_dict.keys():
	camera_dict[i] = np.array(camera_dict[i])

print(camera_dict)
with open(camera_file[:-3]+"pkl", 'wb') as f:
	pickle.dump(camera_dict, f)
