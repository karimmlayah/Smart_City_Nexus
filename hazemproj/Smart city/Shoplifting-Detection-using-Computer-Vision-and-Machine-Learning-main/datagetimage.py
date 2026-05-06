
from roboflow import Roboflow
rf = Roboflow(api_key="54ifGyQXPqel7tKuQzUq")
project = rf.workspace("fastnuces-uakqb").project("fyp-shoplift")
version = project.version(1)
dataset = version.download("yolov11")
                