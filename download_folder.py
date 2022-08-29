import gdown

# download of the Video_Pool folder NOT WORKING

folders = [
    "https://drive.google.com/drive/u/1/folders/18t2Xp0WII6b7tBFHCcHLNv0xshOPNbEY",
    "https://drive.google.com/drive/u/1/folders/1Yx_3SvOQGiphFrI65uIRBe0rPEIWr8Rp",

]

files = {
    "annotations.txt": "https://drive.google.com/drive/u/1/folders/1RKMx3ZssjWnxwU0MUIQPzVp4xOmBUOzq"
}

for folder in folders:
    gdown.download_folder(folder)

for filename, file_url in files.items():
    gdown.download(file_url, filename)