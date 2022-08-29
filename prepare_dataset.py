import os
import pandas as pd
from pathlib import Path
import shutil
from tqdm import tqdm

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@
BALANCED_DATASET = 1
NON_BALANCED_DATASET = 0
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@

MODE = BALANCED_DATASET
dataset = pd.DataFrame()
for file in (x for x in os.listdir("./annotations") if os.path.isfile(f"./annotations/{x}")):
    dataset = dataset.append(pd.read_csv(f'./annotations/{file}', names=["file_name", "label"], delimiter="\t"), ignore_index=True)
# Get the different classes
classes = pd.read_csv("./classes.txt",names=["label","classes"])
dataset = dataset.merge(classes, how="inner")

print(dataset.groupby("label").count()["file_name"])
print(dataset.groupby("label").apply(lambda x: round(x["file_name"].count()/dataset.count() * 100,2))["file_name"])


if MODE == BALANCED_DATASET:
  dataset=dataset.groupby("label",as_index=False).apply(lambda x: x.sample(frac=1)[:100])


training_set = dataset.groupby("label",as_index=False).apply(lambda x: x.sample(frac=1)[:int(((len(x)*60)/100))])

remaining_data_dataset = dataset.drop(training_set.index.get_level_values(1))
validation_set = remaining_data_dataset.groupby("label",as_index=False).apply(lambda x: x.sample(frac=1)[:int(((len(x)*50)/100))])

remaining_data_dataset = remaining_data_dataset.drop(validation_set.index.get_level_values(1))
test_set = remaining_data_dataset.groupby("label",as_index=False).apply(lambda x: x.sample(frac=1)[:int(((len(x)*50)/100))])

shutil.rmtree("./train/")
shutil.rmtree("./validation/")
shutil.rmtree("./test/")

os.makedirs("train")
os.makedirs("validation")
os.makedirs("test")

# Copy videos in the proper set
for file_name in tqdm(training_set.file_name):
  shutil.copy(f"./Video_Pool/{file_name}", f"./train/{file_name}")

for file_name in tqdm(validation_set.file_name):
  shutil.copy(f"./Video_Pool/{file_name}", f"./validation/{file_name}")

for file_name in tqdm(test_set.file_name):
  shutil.copy(f"./Video_Pool/{file_name}", f"./test/{file_name}")

training_set.to_csv("./training.csv", sep=" ", header=False, index=False, columns=["file_name","label"])
validation_set.to_csv("./validation.csv", sep=" ", header=False, index=False, columns=["file_name","label"])
test_set.to_csv("./test.csv", sep=" ", header=False, index=False, columns=["file_name","label"])