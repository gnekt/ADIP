import gdown

# download of the Video_Pool folder NOT WORKING
from sensible_data import folders
from sensible_data import files


for folder in folders:
    gdown.download_folder(folder)

for filename, file_url in files.items():
    gdown.download(file_url, filename)