import pyexr
depth_path = '/home/adam/2tb-workspace/RDNU/M3VIR-extracted/Track4/Residential-Areas/Scene04/MovingCameraStaticScene/Realistic_1920x1080_1024sample/Depth_images/Realistic_1920x1080_1024sample_Back_0000.exr'
file = pyexr.open(depth_path)
depth = file.get()
print('pyexr shape directly from file:', depth.shape)
