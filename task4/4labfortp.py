import argparse
import multiprocessing
import time
import cv2


class Sensor:
    def get(self):
        raise NotImplementedError("Subclasses must implement method get()")


class SensorX(Sensor):
    '''Sensor X'''

    def __init__(self, delay: float):
        self._delay = delay
        self._data = 0

    def get(self) -> int:
        time.sleep(self._delay)
        self._data += 1
        return self._data


class SensorCam(Sensor):
    def __init__(self, cam, res):
        if cam == 'default':
            self.cap = cv2.VideoCapture(0)
        else:
            self.cap = cv2.VideoCapture(cam)
        self.cap.set(3, res[0])
        self.cap.set(4, res[1])

    def get(self):
        ret, frame = self.cap.read()
        return frame

    def __del__(self):
        self.cap.release()


class WindowImage:
    def __init__(self, freq):
        self.freq = freq
        cv2.namedWindow("window")

    def show(self, img, s1, s2, s3):
        x = img.shape[1] - 700
        y = img.shape[0] - 50
        text = f"Sensor 1: {s1} Sensor 2: {s2} Sensor 3: {s3}"
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        cv2.imshow("window", img)

    def __del__(self):
        cv2.destroyWindow("window")


def process(que, sensor):
    while True:
        new_sens = sensor.get()
        if que.empty():
            que.put(new_sens)


def main(args):
    picsize = (int(args.res.split('*')[0]), int(args.res.split('*')[1]))
    sensor1 = SensorX(1)
    sensor2 = SensorX(0.1)
    sensor3 = SensorX(0.01)
    window = WindowImage(args.freq)
    camera = SensorCam(args.cam, picsize)

    que1 = multipr  ocessing.Queue()
    que2 = multiprocessing.Queue()
    que3 = multiprocessing.Queue()

    proc1 = multiprocessing.Process(target=process, args=(que1, sensor1))
    proc2 = multiprocessing.Process(target=process, args=(que2, sensor2))
    proc3 = multiprocessing.Process(target=process, args=(que3, sensor3))

    proc1.start()
    proc2.start()
    proc3.start()

    sens1 = sens2 = sens3 = 0
    while True:
        if not que1.empty():
            sens1 = que1.get()
        if not que2.empty():
            sens2 = que2.get()
        if not que3.empty():
            sens3 = que3.get()
        sensim = camera.get()

        window.show(sensim, sens1, sens2, sens3)
        time.sleep(1 / args.freq)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    proc1.terminate()
    proc2.terminate()
    proc3.terminate()

    proc1.join()
    proc2.join()
    proc3.join()



parser = argparse.ArgumentParser()
parser.add_argument('--cam', type=str, default='default', help='Camera name')
parser.add_argument('--res', type=str, default='900*900', help='Camera resolution')
parser.add_argument('--freq', type=int, default=40, help='Output frequency')
args = parser.parse_args()
main(args)