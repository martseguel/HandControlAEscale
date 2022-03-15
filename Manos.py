import math
import cv2
import mediapipe as mp
import time

class detectormanos():
    def __init__(self, mode=False, maximo_manos = 2, FC_det = 0.5, FC_seg = 0.5):
        self.mode = mode
        self.maximo_manos = maximo_manos
        self.FC_det = FC_det
        self.FC_seg = FC_seg

        self.mpManos = mp.solutions.hands
        self.manos = self.mpManos.Hands(self.mode, self.maximo_manos, self.FC_det, self.FC_seg)
        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4,8,12,16,20]

    def findHands(self, frame, draw = True):
        imgcolor = cv2.cvtColor(frame ,cv2.COLOR_BGR2RGB)
        self.results = self.manos.process(imgcolor)

        if self.results.multi_hand_landmarks:
            for mano in self.results.multi_hand_landmarks:
                if draw:
                    self.dibujo.draw_landmarks(frame, mano, self.mpManos.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, numMano = 0, draw = True):
        xList = []
        yList = []
        bbox = []
        self.list = []
        if self.results.multi_hand_landmarks:
            miMano = self.results.multi_hand_landmarks[numMano]
            for id, lm in enumerate(miMano.landmark):
                alto, ancho, c = frame.shape
                cx, cy = int(lm.x * ancho), int(lm.y * alto)
                xList.append(cx)
                yList.append(cy)
                self.list.append([id,cx,cy])
                if draw:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

            xMin, xMax = min(xList), max(xList)
            yMin, yMax = min(yList), max(yList)
            bbox = xMin, yMin, xMax, yMax
            if draw:
                cv2.rectangle(frame, (xMin -20, yMin -20), (xMax +20, yMax +20), (0, 255, 0), 2)
        return self.list, bbox

    def fingers(self):
        fings = []
        if self.list[self.tip[0]][1] > self.list[self.tip[0]-1][1]:
            fings.append(1)
        else:
            fings.append(0)

        for id in range(1,5):
            if self.list[self.tip[id]][2] < self.list[self.tip[id]-2][2]:
                fings.append(1)
            else:
                fings.append(0)
        return fings

    def distance(self, p1, p2, frame, draw = True, r=15, t=3):
        x1, y1 = self.list[p1][1:]
        x2, y2 = self.list[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if draw:
            cv2.line(frame, (x1,y1), (x2,y2), (0,0,255), t)
            cv2.circle(frame, (x1,y1), r, (0,0,255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        length = math.hypot(x2-x1, y2-y1)
        return length, frame, [x1,y1,x2,y2,cx,cy]

def main():
    ptime = 0
    ctime = 0

    cap = cv2.VideoCapture(0)
    detector = detectormanos()

    while True:
        ret, frame = cap.read()
        frame = detector.findHands(frame)
        list, bbox = detector.findPosition(frame)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(frame, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow("Manos", frame)
        k = cv2.waitKey(1)
        if k == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
