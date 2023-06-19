from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QApplication, QFileDialog
from tkinter import Tk

from control.Control import MainFunctions, GenerateButtonsEvent, TrajectoryWorker
from control.Tab.TabCompare.TabCompare_2 import TabCompare2
from control.Tab.TabTest.TabTest import TabTest
from control.Tab.TabSpot.Tab3 import Tab3
from control.Tab4 import Tab4
from control.Tab.TabCarInspection.TabCarInspection import TabCarInspection
from control.Tab.TabCollectionData.TabCollectionData import TabCollectionData
from control.Tab.TabCompare.TabCompare import TabCompare
from spot.SpotRobot import Robot
from view.Main_UI import Ui_MainWindow

WIDTH  = Tk().winfo_screenwidth()
HEIGHT = Tk().winfo_screenheight()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_window = Ui_MainWindow()
        self.main_window.setupUi(self)
        # self.setWindowFlags(Qt.WindowStaysOnTopHint)
        # self.setWindowFlag(Qt.FramelessWindowHint)
        # self.setGeometry(0, 0, WIDTH, HEIGHT)
        # self.setMinimumSize(int(WIDTH * 0.8), int(HEIGHT * 0.8))

        self.file_dialog = QFileDialog()
        self.file_dialog.setNameFilters(["Text files (*.txt)", "Images (*.png *.jpg)"])
        self.file_dialog.selectNameFilter("Images (*.png *.jpg)")
        self.main_functions = MainFunctions(self)

        self.robot = Robot()
        self.robot_commander = None

        self.tab_test = TabTest(self)
        self.tab3 = Tab3(self)
        self.tab4 = Tab4(self)
        self.tab_data_collection = TabCollectionData(self)
        self.tab_car_inspection = TabCarInspection(self)
        # self.tab_compare = TabCompare(self)
        self.tab_compare = TabCompare2(self)
        self.trajectory_worker = TrajectoryWorker(self, self.main_window, self.tab_test)
        self.trajectory_worker.finished.connect(self.tab_test.page5.work_fisished_event)

        self.main_window.tabMain.currentChanged.connect(self.check_robot_connection)

    def __del__(self):
        print("end program")

    def check_robot_connection(self):
        if self.main_window.tabMain.currentIndex() == 0:
            return

        # if self.robot.robot is None:
        #     self.main_functions.show_message_box("로봇 연결이 필요합니다.")
        #     self.main_window.tabMain.setCurrentIndex(0)
        #     return
        #
        # if self.robot._lease_keepalive is None:
        #     self.main_functions.show_message_box("제어권(LEASE)을 활성화 시켜야 합니다.")
        #     self.main_window.tabMain.setCurrentIndex(0)
        #     return

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        close = QMessageBox()
        close.setWindowFlags(Qt.WindowStaysOnTopHint)
        close.setIcon(QMessageBox.Icon.Critical)
        close.setWindowTitle("종료")
        close.setText("종료하시겠습니까?")
        close.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        close = close.exec()

        if close == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()

    def event(self, event):
        if isinstance(event, GenerateButtonsEvent):
            self.tab_test.page5.generate_buttons(event.iteration)
            return True
        return super().event(event)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
