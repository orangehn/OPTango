# Prerequests
- linux ida
- IDA Python (python3) with networkx, pyelftools, binaryai

# Run in Windows
Install env
```shell
conda create -n env_name python=3.8
conda activate env_name
# https://pytorch.org/get-started/previous-versions/
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch

pip install tqdm
pip install pyelftools binaryai networkx
pip install yara
# modify code split("/")[-1] to os.path.split()[-1]
# mov ida to projects\jTrans\datautils\ida
```

Run 
```shell
cd projects/jTrans/datautils
conda activate env_name
"""
Fatal Python error: init_fs_encoding: failed to get the Python codec of the filesystem encoding
Python runtime state: core initialized
ModuleNotFoundError: No module named 'encodings'
"""
set PYTHONPATH=D:\software\miniconda3\envs\env_name
set PYTHONHOME=D:\software\miniconda3\envs\env_name
.\\ida\\idapyswitch.exe --force-path D:\software\miniconda3\envs\env_name\python3.dll

conda activate mmdetection
cd projects/jTrans/datautils
set PYTHONPATH=D:\IDE\AnaConda\envs\mmdetection
set PYTHONHOME=D:\IDE\AnaConda\envs\mmdetection
.\\ida\\idapyswitch.exe --force-path D:\IDE\AnaConda\envs\mmdetection\python3.dll

python run.py
```