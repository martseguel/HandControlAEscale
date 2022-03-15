import cv2
import Manos as sm
import numpy as np

from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

anchoCam, altoCam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, anchoCam)
cap.set(4, altoCam)

detector = sm.detectormanos(maximo_manos=1, FC_det=0.7)

dispositivos = AudioUtilities.GetSpeakers()
interfaz = dispositivos.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volumen = cast(interfaz, POINTER(IAudioEndpointVolume))
RangeVol = volumen.GetVolumeRange()
print(RangeVol)
VolMin = RangeVol[0]
VolMax = RangeVol[1]

while True:
    ret, frame = cap.read()
    frame = detector.findHands(frame)
    list, bbox = detector.findPosition(frame, draw=False)
    if len(list) != 0:
        x1, y1 = list[4][1], list[4][2]
        x2, y2 = list[8][1], list[8][2]

        fings = detector.fingers()

        if fings[0] == 1 and fings[1] == 1:
            longitud, frame, linea = detector.distance(4, 8, frame, r=8, t=2)
            print(longitud)

            vol = np.interp(longitud, [25, 100], [VolMin, VolMax])
            volumen.SetMasterVolumeLevel(vol, None)

            if longitud<25:
                cv2.circle(frame, (linea[4], linea[5]), 10, (0,255,0), cv2.FILLED)

    cv2.imshow("Video", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break

cap.release()
cv2.destroyAllWindows()
