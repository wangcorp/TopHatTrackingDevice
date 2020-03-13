from picamera import PiCamera
from time import sleep

camera=PiCamera()
#camera.resolution=(1920,1080)
camera.resolution=(1296,730)
#camera.resolution=(680,480)

camera.start_preview()

try:
    while True:
        pass
except KeyboardInterrupt:
    
    camera.stop_preview()    