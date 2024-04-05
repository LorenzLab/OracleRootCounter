# Import Google libraries
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive

import os
import sys

LABELS_FOLDER_ID = "1QVJ_jlhVe3HGL5G2C6yrOh5zM1DrVOIW"
IMAGES_FOLDER_ID = "19tNxmg48JePXX9VDyvuQrSHaGOE1-VAQ"

id_lists = os.listdir("dataset_out/labels")
id_lists = [i.split('.')[0] for i in id_lists]

# split the id_lists into two parts
part1 = id_lists[:len(id_lists) // 2]
part2 = id_lists[len(id_lists) // 2:]

gauth = GoogleAuth()
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.
drive = GoogleDrive(gauth)

# file_list = drive.ListFile({'q': "'1XPNJXc4IvOMumELLHTIbxIqvngLiyNQD' in parents and trashed=false"}).GetList()
# for file1 in file_list:
#     print('title: %s, id: %s' % (file1['title'], file1['id']))

for id_ in part1:
    file = drive.CreateFile({"parents": [{"id": LABELS_FOLDER_ID}], "title": f"{id_}.txt", "mimeType": "text/plain"})
    file.SetContentFile(f"dataset_out/labels/{id_}.txt")
    file.Upload()
    file = drive.CreateFile({"parents": [{"id": IMAGES_FOLDER_ID}], "title": f"{id_}.png", "mimeType": "image/png"})
    file.SetContentFile(f"dataset_out/images/{id_}.png")
    file.Upload()
