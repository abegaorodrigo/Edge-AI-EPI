from roboflow import Roboflow
rf = Roboflow(api_key="5MYSxXJiH8m0n42KHBBb")
project = rf.workspace("hx-hezqh").project("ppe-detection-yfmym")
version = project.version(1)
dataset = version.download("yolov8-obb")
