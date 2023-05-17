import os
import sys
import time
import uuid
import numpy as np
from datetime import datetime

import cv2
from loguru import logger
from PyQt5 import uic
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QUrl
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent


class MainWindow(QMainWindow):
    stream_on = False
    alert_on = False
    model_prediction = False
    sent_messages = False

    # Load class list
    classes = []
    with open("./dnn_model/classes.txt", "rt") as f:
        for class_name in f.readlines():
            class_name = class_name.strip()
            classes.append(class_name)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Intellgent Video Surveilece")

        self.ui = uic.loadUi("stream.ui", self)
        self.setFixedSize(self.size())
        self.ui.start_stream.clicked.connect(self.handle_start_stream)
        self.ui.stop_stream.clicked.connect(self.handle_stop_stream)
        self.ui.clear_logs.clicked.connect(self.handle_clear_update)

    def handle_start_stream(self):
        if not MainWindow.stream_on:
            self.worker_stream = WorkerStream()
            self.worker_stream.start()
            MainWindow.stream_on = True
            self.worker_stream.image_update.connect(self.handle_to_predictor)
            logger.info("Stream thread started")

        if not MainWindow.model_prediction:
            self.worker_predictor = WorkerPredictor()
            self.worker_predictor.start()
            MainWindow.model_prediction = True
            self.worker_predictor.predicted.connect(self.handle_image_update)
            logger.info("Prediction thread started")

        if not MainWindow.alert_on:
            MainWindow.alert_on = True
            self.worker_alert = WorkerAlert()
            self.worker_alert.start()
            logger.info("Alert thread started")

        if not MainWindow.sent_messages:
            MainWindow.sent_messages = True
            self.worker_message = WorkerMessage()
            self.worker_message.start()
            logger.info("Email and SMS thread started")

    def handle_stop_stream(self):
        if MainWindow.stream_on:
            MainWindow.stream_on = False
            self.worker_stream.stop()
            logger.info("Stream thread stopped")

        if MainWindow.model_prediction:
            MainWindow.model_prediction = False
            self.worker_predictor.stop()
            logger.info("Prediction thread stopped")

        if MainWindow.alert_on:
            MainWindow.alert_on = False
            self.worker_alert.stop()
            logger.info("Alert Thread stopped")

        if MainWindow.sent_messages:
            MainWindow.sent_messages = False
            self.worker_message.stop()
            logger.info("Email and SMS Thread stopped")

    def handle_to_predictor(self, frame):
        self.worker_predictor.predictor_frame = np.copy(frame)

    def handle_image_update(self, result):
        frame = np.copy(result[0])
        model_fps = result[-1]
        model_fps = round(model_fps, 2)
        model_fps = "model_fps: " + str(model_fps)
        params = {}
        min_score = 1
        min_score_obj = None
        if len(result) > 2:
            logger.critical("weapons detected")
            self.worker_alert.play_alert = True
            class_ids, scores, bboxes = result[1]
            for class_id, score, bbox in zip(class_ids, scores, bboxes):
                if self.classes[class_id] in params.keys():
                    params[self.classes[class_id]][0] = round(
                        (score * 100 + params[self.classes[class_id]][0]) / 2, 2
                    )
                    params[self.classes[class_id]][1] += 1
                else:
                    params[self.classes[class_id]] = []
                    params[self.classes[class_id]].append(round(score * 100))
                    params[self.classes[class_id]].append(1)

                if score < min_score:
                    min_score = score
                    min_score_obj = class_id, score, bbox
                x, y, w, h = bbox
                cv2.putText(
                    frame,
                    MainWindow.classes[class_id],
                    (x, y - 10),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (0, 0, 255),
                    1,
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            self.update_logs(params=params)

            # save images
            if len(os.listdir("images")) < 20:
                if not hasattr(self, "file_path"):
                    file_path = "images/" + str(uuid.uuid4()) + " FULL.jpg"
                cv2.imwrite(file_path, frame)

        ######## Main Frame ########
        cv2.putText(
            frame, model_fps, (280, 15), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1
        )
        frame = QImage(
            frame.data,
            frame.shape[1],
            frame.shape[0],
            3 * frame.shape[1],
            QImage.Format.Format_RGB888,
        ).rgbSwapped()
        self.ui.frame_holder.setPixmap(QPixmap.fromImage(frame))

        ######## Zoomed Frame ########
        if min_score_obj is not None:
            x, y, w, h = min_score_obj[-1]
            frame = result[0][y : y + h, x : x + w]
            file_path = "images/" + self.current_time + " ZOOMED.jpg"
            cv2.imwrite(file_path, frame)
            frame = cv2.resize(frame, (200, 190))
            frame = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                3 * frame.shape[1],
                QImage.Format.Format_RGB888,
            ).rgbSwapped()
            self.ui.zoom.setPixmap(QPixmap.fromImage(frame))

            WorkerMessage.params = params

        else:
            frame = result[0]
            frame = cv2.resize(frame, (200, 190))
            frame = QImage(
                frame.data,
                frame.shape[1],
                frame.shape[0],
                3 * frame.shape[1],
                QImage.Format.Format_RGB888,
            ).rgbSwapped()
            self.ui.zoom.setPixmap(QPixmap.fromImage(frame))

        ######## Black-White Frame ########
        frame = result[0]
        tmp = np.zeros(shape=(416, 416, 3), dtype=np.int8)
        if len(result) > 2:
            _, _, bboxes = result[1]
            tmp_cpy = np.copy(tmp)
            for box in bboxes:
                x, y, w, h = box
                cv2.rectangle(tmp_cpy, (x, y), (x + w, y + h), (255, 255, 255), -1)
            tmp_cpy = QImage(
                tmp_cpy.data,
                tmp_cpy.shape[1],
                tmp_cpy.shape[0],
                3 * tmp_cpy.shape[1],
                QImage.Format.Format_RGB888,
            ).rgbSwapped()
            tmp_cpy = tmp_cpy.scaled(200, 190)
            self.ui.threat_area.setPixmap(QPixmap.fromImage(tmp_cpy))
            return
        tmp = QImage(
            tmp.data,
            tmp.shape[1],
            tmp.shape[0],
            3 * tmp.shape[1],
            QImage.Format.Format_RGB888,
        ).rgbSwapped()
        tmp = tmp.scaled(200, 190)
        self.ui.threat_area.setPixmap(QPixmap.fromImage(tmp))

    def update_logs(self, params={}) -> None:
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not params:
            return
        txt = f"### {self.current_time} ###"
        for idx, key in enumerate(params.keys()):
            avg_accuracy, numbers = params[key]
            tmp = f"""
{idx+1}.  {key}:
        - Accuracy:\t{avg_accuracy} %
        - Numbers :\t{numbers}"""
            txt += tmp

        txt += "\n####################\n\n\n"

        prev_txt = self.ui.logs.text()
        self.ui.logs.setText(txt + prev_txt)
        self.ui.logs.setAlignment(Qt.AlignLeft | Qt.AlignTop)

    def handle_clear_update(self):
        self.ui.logs.setText("")


class WorkerStream(QThread):
    # Custom signal
    image_update = pyqtSignal(np.ndarray)

    FONT = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 1
    FONT_THICKNESS = 1

    def run(self):
        self.ThreadActive = True
        cap = cv2.VideoCapture(0)

        while self.ThreadActive:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.resize(frame, (416, 416))
            self.image_update.emit(frame)
        cap.release()

    def stop(self):
        self.ThreadActive = False


class WorkerPredictor(QThread):
    predicted = pyqtSignal(tuple)

    ############################ LOAD MODEL ############################
    net = cv2.dnn.readNet(
        "/home/deshrit/ssd/major_project/final_demo/intelligent_video_surveillence/dnn_model/yolov4-tiny-custom_best.weights", 
        "/home/deshrit/ssd/major_project/final_demo/intelligent_video_surveillence/dnn_model/yolov4-tiny-custom.cfg"
    )
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1 / 255)
    ####################################################################

    def run(self):
        self.predictor_frame = None
        self.ThreadActive = True
        prev_time = time.time()
        while self.ThreadActive:
            if self.predictor_frame is not None:
                model_result = WorkerPredictor.model.detect(self.predictor_frame)
                current_time = time.time()
                diff_time = current_time - prev_time
                prev_time = current_time
                model_fps = 1 / (diff_time)
                if isinstance(model_result[0], np.ndarray):
                    self.predicted.emit((self.predictor_frame, model_result, model_fps))
                    self.frame = None
                    continue
                self.predicted.emit((self.predictor_frame, model_fps))
                self.predictor_frame = None

    def stop(self):
        self.ThreadActive = False


class WorkerAlert(QThread):
    play_alert = False
    state = 0

    def run(self) -> None:
        self.ThreadActive = True

        self.player = QMediaPlayer()
        file_path = os.path.join(os.getcwd(), "audios/alert.mp3")
        url = QUrl.fromLocalFile(file_path)
        content = QMediaContent(url)
        self.player.setMedia(content)
        while self.ThreadActive:
            if self.play_alert == True:
                self.ring()
            #     continue
            # self.stop_ring()

    def ring(self):
        if self.player.state() == 0:
            self.player.play()

    def stop_ring(self):
        self.play_alert == False
        if self.player.state() == 1:
            self.player.stop()

    def stop(self):
        self.player.stop()
        self.ThreadActive = False


class WorkerMessage(QThread):
    params_signal = pyqtSignal()
    params = None

    def run(self) -> None:
        self.ThreadActive = True

        import json
        from mail import mail
        from sms import sms

        # Load credentials
        with open("credentials.json", "rt") as f:
            credentials = json.load(f)

        while self.ThreadActive:
            if WorkerMessage.params:
                mail.send_mail(credentials["mail"], WorkerMessage.params)
                sms.send_sms(credentials["sms"], WorkerMessage.params)
                WorkerMessage.params = None
                MainWindow.sent_messages = True
                time.sleep(0.1)

    def stop(self) -> None:
        self.ThreadActive = False


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
