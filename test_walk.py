import os

top_dir = "Datasets"
dataset = "ETL_test"
root_dir = os.path.join(top_dir, dataset)
images_name = []
for path, subdir, files in os.walk(root_dir):
    images_name.append(path)
    print(files)