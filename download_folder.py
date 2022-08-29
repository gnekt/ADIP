
import gdown

# download of the Video_Pool folder NOT WORKING
from sensible_data import dataset
from sensible_data import annotation


for file in dataset:
    for filename, file_url in file.items():
        gdown.download(url=file_url, output=filename, quiet=False, fuzzy=True)

for filename, file_url in annotation.items():
    gdown.download_folder(file_url, quiet=False, use_cookies=False)