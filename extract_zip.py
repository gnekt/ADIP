import zipfile

files_to_extract = [
    "Video_Pool-20220827T142731Z-001.zip",
    "Video_Pool-20220827T142731Z-002.zip"
]

for filename in files_to_extract:
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall("./Video_Pool")