import os,sys
import logging
from pathlib import Path

while True:
    project_name=input('Enter your projrct name')
    if project_name != "":
        break

list_of_files=[
    f"{project_name}/__init__.py",
    f"{project_name}/components/__init__.py",
    f"{project_name}/config/__init__.py",
    f"{project_name}/constants/__init__.py",
    f"{project_name}/entity/__init__.py",
    f"{project_name}/exception/__init__.py",
    f"{project_name}/logger/__init__.py",
    f"{project_name}/pipeline/__init__.py",
    f"{project_name}/Utils/__init__.py"
    f"config/confi.yaml",
    "schema.yaml",
    'app.py',
    "logs.py",
    "exception.py",
    "setup.py",
    "requirements.txt",
    "pipeline.txt"


]

for file_path in list_of_files:
    file_path=Path(file_path)
    filedir,file_name=os.path.split(file_path)
    if filedir != "":
        os.makedirs(filedir,exist_ok=True)
    if (not os.path.exists(file_path) or os.path.getsize(file_path==0)):
        with open(file_path,'w') as f:
            pass

    else:
        logging.info("{file_name} already exists at {file_path}")

