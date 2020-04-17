from argparse import ArgumentParser
import cv2
import sys
import time
import numpy as np
import threading
from imutils.video.filevideostream import FileVideoStream
from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue
# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue


print("OpenCV version: {}".format(cv2.__version__))
# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()


CNN_INPUT_SIZE = 128


# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default="abc.mp4",
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()


class Main(threading.Thread):

    def __init__(self, cap, box_que, img_que, detector, video_src, sample_frame):
        threading.Thread.__init__(self)
        # self.condition = condition
        self.cap = cap
        self.img_que = img_que
        self.box_que = box_que
        self.mark_detector = detector
        self.video_src = video_src
        self.sample_frame = sample_frame

    def get_face(self, detector, img_queue, box_queue):
        """Get face from image queue. This function is used for multiprocessing"""
        while True:
            image = img_queue.get()
            box = detector.extract_cnn_facebox(image)
            box_queue.put(box)

    def run(self):

        # Introduce mark_detector to detect landmarks.
        box_thread = threading.Thread(target=self.get_face,
                                      args=(self.mark_detector, self.img_que, self.box_que))
        box_thread.start()

        head_thread = threading.Thread(target=self.process_head_pose,
                                       args=(self.sample_frame, self.img_que, self.box_que,
                                             self.cap, self.video_src, self.mark_detector))
        head_thread.start()

    def process_head_pose(self, sample_frame, img_queue, box_queue, cap, video_src, mark_detector):
        height, width = sample_frame.shape[:2]
        pose_estimator = PoseEstimator(img_size=(height, width))

        # Introduce scalar stabilizers for pose.
        pose_stabilizers = [Stabilizer(
            state_num=2,
            measure_num=1,
            cov_process=0.1,
            cov_measure=0.1) for _ in range(6)]

        tm = cv2.TickMeter()
        # cap.start()
        while True:
            # Read frame, crop it, flip it, suits your needs.
            frame = cap.read()
            if frame is None:
                break

            # Crop it if frame is larger than expected.
            # frame = frame[0:480, 300:940]

            # If frame comes from webcam, flip it so it looks like a mirror.
            if video_src == 0:
                frame = cv2.flip(frame, 2)

            # Pose estimation by 3 steps:
            # 1. detect face;
            # 2. detect landmarks;
            # 3. estimate pose

            # Feed frame to image queue.
            img_queue.put(frame)

            # Get face from box queue.
            facebox = box_queue.get()

            if facebox is not None:
                # Detect landmarks from image of 128x128.
                face_img = frame[facebox[1]: facebox[3],
                           facebox[0]: facebox[2]]
                face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
                face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

                tm.start()
                marks = mark_detector.detect_marks([face_img])
                tm.stop()

                # Convert the marks locations from local CNN to global image.
                marks *= (facebox[2] - facebox[0])
                marks[:, 0] += facebox[0]
                marks[:, 1] += facebox[1]

                # Uncomment following line to show raw marks.
                # mark_detector.draw_marks(
                #     frame, marks, color=(0, 255, 0))

                # Uncomment following line to show facebox.
                # mark_detector.draw_box(frame, [facebox])

                # Try pose estimation with 68 points.
                pose = pose_estimator.solve_pose_by_68_points(marks)

                # Stabilize the pose.
                steady_pose = []
                pose_np = np.array(pose).flatten()
                for value, ps_stb in zip(pose_np, pose_stabilizers):
                    ps_stb.update([value])
                    steady_pose.append(ps_stb.state[0])
                steady_pose = np.reshape(steady_pose, (-1, 3))

                # Uncomment following line to draw pose annotation on frame.
                # pose_estimator.draw_annotation_box(
                #     frame, pose[0], pose[1], color=(255, 128, 128))

                # Uncomment following line to draw stabile pose annotation on frame.
                pose_estimator.draw_annotation_box(
                    frame, steady_pose[0], steady_pose[1], color=(128, 255, 128))

                # Uncomment following line to draw head axes on frame.
                # pose_estimator.draw_axes(frame, stabile_pose[0], stabile_pose[1])

            # Show preview.
            cv2.imshow("Preview", frame)
            if cv2.waitKey(1) == 27:
                break

    def stop(self):
        self.join()


def main(): 
    # Video source from webcam or video file.

    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = FileVideoStream(video_src, queue_size=200)
    cap.start()
    # time.sleep(0.15)
    if video_src == 0:
        cap.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    sample_frame = cap.read()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    mark_detector = MarkDetector()
    box_process = Main(cap, box_queue, img_queue, mark_detector, video_src, sample_frame)
    box_process.start()
    # cap.stop()


if __name__ == '__main__':
    # Video source from webcam or video file.
    main()
