import zipfile

files_to_extract = [
    "Video_Pool1.zip",
    "Video_Pool2.zip"
]

for filename in files_to_extract:
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall("./Video_Pool")