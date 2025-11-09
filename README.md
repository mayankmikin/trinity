# trinity
yolo_live_object_detection_multistream

### create e new virtual environment
> python3 -m venv trinity

#####
Here, trinity is the name of the directory where the virtual environment will be created. You can choose any name you like.

#### CMDS 
> source trinity/bin/activate
> deactivate

#### to remove the virtual environment 
> rm -rf <virtual_environment_name>

#### PIP - libraries install and setup 
###### SAVE in requirement .txt
> pip freeze > requirements.txt
##### install packages befofe deployment/local environment command 
> pip install -r requirements.txt
> pip freeze > requirements.txt 

### install packages 
> pip install opencv-python