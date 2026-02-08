import sys

input_dir = sys.argv[1]
output_dir = sys.argv[2]


f = open("configs/run/demo.yaml")

file_content = f.read().split("\n")
for i in range(len(file_content)):
	if("input_dir:" in file_content[i]):
		file_content[i] ="input_dir: \'" + input_dir + "\'"
	if("output_dir:" in file_content[i]):
                file_content[i] ="output_dir: \'" + output_dir + "\'"

file_content_string = ""	
for i in range(len(file_content)):
	file_content_string = file_content_string + file_content[i] +"\n"


with open("configs/run/demo.yaml", "w") as f:
  f.write(file_content_string)
