import OpenEXR
import Imath
import numpy as np

def read_exr_bytes(content):
    with open("temp.exr", "wb") as f:
        f.write(content)
    
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    exr_file = OpenEXR.InputFile("temp.exr")
    dw = exr_file.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    
    channels = exr_file.header()['channels'].keys()
    # Read channels
    color_channels = ['R', 'G', 'B'] if 'R' in channels else [list(channels)[0]]
    
    img = []
    for c in color_channels:
        img.append(np.frombuffer(exr_file.channel(c, pt), dtype=np.float32).reshape(size[1], size[0]))
    
    img = np.stack(img, axis=-1)
    return img

print("OpenEXR test ok")
