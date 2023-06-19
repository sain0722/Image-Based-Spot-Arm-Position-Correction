from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QDialog


class ParameterDialog(QDialog):
    def __init__(self):
        super().__init__()
        # Main dialog setup
        self.setWindowFlags(Qt.WindowStaysOnTopHint)
        self.setObjectName("Parameter Setting")
        self.resize(519, 533)
        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setObjectName("verticalLayout")

        self.font = QtGui.QFont()
        self.font.setFamily("현대하모니 M")
        self.font.setPointSize(10)

        # Setup "Depth 누적 횟수"
        self.lbl_depth_acm_count = QtWidgets.QLabel(self)
        self.sbx_depth_acm_count = QtWidgets.QSpinBox(self)
        self.setupFirstSection()

        # Setup "Outlier 제거 파라미터"
        self.verticalFrame = QtWidgets.QFrame(self)
        self.gridFrame_iqr = QtWidgets.QFrame(self.verticalFrame)

        self.lbl_outlier_remove_parameters = QtWidgets.QLabel(self)
        self.verticalFrame = QtWidgets.QFrame(self)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalFrame)
        self.cbx_iqr = QtWidgets.QCheckBox(self.verticalFrame)
        self.gridFrame_iqr = QtWidgets.QFrame(self.verticalFrame)

        self.horizontalFrame = QtWidgets.QFrame(self.verticalFrame)
        self.gridLayout = QtWidgets.QGridLayout(self.gridFrame_iqr)
        self.lbl_iqr1 = QtWidgets.QLabel(self.gridFrame_iqr)
        self.lbl_iqr3 = QtWidgets.QLabel(self.gridFrame_iqr)
        self.sbx_iqr1 = QtWidgets.QSpinBox(self.gridFrame_iqr)
        self.sbx_iqr3 = QtWidgets.QSpinBox(self.gridFrame_iqr)
        self.cbx_gaussian = QtWidgets.QCheckBox(self.verticalFrame)
        self.gridFrame_outlier = QtWidgets.QFrame(self.verticalFrame)

        self.sbx_std_ratio = QtWidgets.QDoubleSpinBox(self.gridFrame_outlier)
        self.lbl_std_ratio = QtWidgets.QLabel(self.gridFrame_outlier)
        self.lbl_nb_neighbors = QtWidgets.QLabel(self.gridFrame_outlier)
        self.sbx_nb_neighbors = QtWidgets.QSpinBox(self.gridFrame_outlier)
        self.gridLayout_2 = QtWidgets.QGridLayout(self.gridFrame_outlier)
        self.cbx_sor_filter = QtWidgets.QCheckBox(self.verticalFrame)
        self.sbx_gaussian_threshold = QtWidgets.QDoubleSpinBox(self.horizontalFrame)
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.horizontalFrame)
        self.lbl_gaussian_threshold = QtWidgets.QLabel(self.horizontalFrame)

        self.setupSecondSection()

        # Setup "ICP 파라미터"
        self.lbl_icp_parameters = QtWidgets.QLabel(self)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.gridFrame_icp = QtWidgets.QFrame(self)
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridFrame_icp)
        self.lbl_loss_sigma = QtWidgets.QLabel(self.gridFrame_icp)
        self.lbl_icp_threshold = QtWidgets.QLabel(self.gridFrame_icp)
        self.lbl_icp_iteration = QtWidgets.QLabel(self.gridFrame_icp)
        self.sbx_icp_iteration = QtWidgets.QSpinBox(self.gridFrame_icp)
        self.sbx_icp_threshold = QtWidgets.QDoubleSpinBox(self.gridFrame_icp)
        self.sbx_loss_sigma = QtWidgets.QDoubleSpinBox(self.gridFrame_icp)

        self.setupThirdSection()

        # Setup "button box"
        self.buttonBox = QtWidgets.QDialogButtonBox(self)
        self.setupFinalSection()

        self.retranslateUi()

    # First section setup
    def setupFirstSection(self):
        self.lbl_depth_acm_count.setFont(self.font)
        self.lbl_depth_acm_count.setObjectName("lbl_depth_acm_count")
        self.verticalLayout.addWidget(self.lbl_depth_acm_count)

        self.sbx_depth_acm_count.setFont(self.font)
        self.sbx_depth_acm_count.setProperty("value", 16)
        self.sbx_depth_acm_count.setObjectName("sbx_depth_acm_count")
        self.verticalLayout.addWidget(self.sbx_depth_acm_count)

    # Second section setup
    def setupSecondSection(self):
        self.lbl_outlier_remove_parameters.setFont(self.font)
        self.lbl_outlier_remove_parameters.setObjectName("lbl_outlier_remove_parameters")
        self.verticalLayout.addWidget(self.lbl_outlier_remove_parameters)
        self.verticalFrame.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.verticalFrame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.verticalFrame.setLineWidth(1)
        self.verticalFrame.setObjectName("verticalFrame")
        self.verticalLayout_2.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_2.setObjectName("verticalLayout_2")

        self.cbx_iqr.setFont(self.font)
        self.cbx_iqr.setObjectName("cbx_iqr")
        self.cbx_iqr.setChecked(True)
        self.verticalLayout_2.addWidget(self.cbx_iqr)

        self.verticalLayout_2.addWidget(self.gridFrame_iqr)

        self.gridFrame_iqr.setFont(self.font)
        self.gridFrame_iqr.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.gridFrame_iqr.setFrameShadow(QtWidgets.QFrame.Plain)
        self.gridFrame_iqr.setObjectName("gridFrame_iqr")
        self.gridLayout.setContentsMargins(10, 10, 10, 10)
        self.gridLayout.setObjectName("gridLayout")
        self.sbx_iqr1.setProperty("value", 20)
        self.sbx_iqr1.setObjectName("sbx_iqr1")
        self.gridLayout.addWidget(self.sbx_iqr1, 0, 1, 1, 1)
        self.lbl_iqr3.setObjectName("lbl_iqr3")
        self.gridLayout.addWidget(self.lbl_iqr3, 1, 0, 1, 1)
        self.sbx_iqr3.setProperty("value", 80)
        self.sbx_iqr3.setObjectName("sbx_iqr3")
        self.gridLayout.addWidget(self.sbx_iqr3, 1, 1, 1, 1)
        self.lbl_iqr1.setObjectName("lbl_iqr1")
        self.gridLayout.addWidget(self.lbl_iqr1, 0, 0, 1, 1)

        self.cbx_gaussian.setFont(self.font)
        self.cbx_gaussian.setObjectName("cbx_gaussian")
        tooltip = "<html>" \
                  "<head/>" \
                  "<body>" \
                  "<p>데이터 포인트가 가우시안 분포의 평균에서 Threshold 배 이상 떨어져 있다면 그것을 아웃라이어로 간주하고 이를 제거</p>" \
                  "</body>" \
                  "</html>"
        self.cbx_gaussian.setToolTip(tooltip)
        self.cbx_gaussian.setChecked(True)

        self.verticalLayout_2.addWidget(self.cbx_gaussian)

        self.horizontalFrame.setFont(self.font)
        self.horizontalFrame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.horizontalFrame.setObjectName("horizontalFrame")
        self.horizontalLayout.setContentsMargins(10, 10, 10, 10)
        self.horizontalLayout.setObjectName("horizontalLayout")

        self.lbl_gaussian_threshold.setObjectName("lbl_gaussian_threshold")
        self.horizontalLayout.addWidget(self.lbl_gaussian_threshold)

        self.sbx_gaussian_threshold.setSingleStep(0.1)
        self.sbx_gaussian_threshold.setProperty("value", 3.0)
        self.sbx_gaussian_threshold.setObjectName("sbx_gaussian_threshold")
        self.horizontalLayout.addWidget(self.sbx_gaussian_threshold)
        self.verticalLayout_2.addWidget(self.horizontalFrame)

        self.cbx_sor_filter.setFont(self.font)
        self.cbx_sor_filter.setObjectName("cbx_sor_filter")
        self.cbx_sor_filter.setChecked(True)
        self.verticalLayout_2.addWidget(self.cbx_sor_filter)

        self.gridFrame_outlier.setFont(self.font)
        self.gridFrame_outlier.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.gridFrame_outlier.setFrameShadow(QtWidgets.QFrame.Plain)
        self.gridFrame_outlier.setObjectName("gridFrame_outlier")

        self.gridLayout_2.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_2.setObjectName("gridLayout_2")

        self.sbx_nb_neighbors.setProperty("value", 20)
        self.sbx_nb_neighbors.setObjectName("sbx_nb_neighbors")
        self.gridLayout_2.addWidget(self.sbx_nb_neighbors, 0, 1, 1, 1)

        self.lbl_nb_neighbors.setObjectName("lbl_nb_neighbors")
        self.gridLayout_2.addWidget(self.lbl_nb_neighbors, 0, 0, 1, 1)

        self.lbl_std_ratio.setObjectName("lbl_std_ratio")
        self.gridLayout_2.addWidget(self.lbl_std_ratio, 1, 0, 1, 1)

        self.sbx_std_ratio.setSingleStep(0.1)
        self.sbx_std_ratio.setProperty("value", 2.0)
        self.sbx_std_ratio.setObjectName("sbx_std_ratio")
        self.gridLayout_2.addWidget(self.sbx_std_ratio, 1, 1, 1, 1)
        self.verticalLayout_2.addWidget(self.gridFrame_outlier)
        self.verticalLayout.addWidget(self.verticalFrame)

    # Third section setup
    def setupThirdSection(self):
        self.lbl_icp_parameters.setFont(self.font)
        self.lbl_icp_parameters.setObjectName("lbl_icp_parameters")
        self.verticalLayout.addWidget(self.lbl_icp_parameters)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")

        self.gridFrame_icp.setFont(self.font)
        self.gridFrame_icp.setFrameShape(QtWidgets.QFrame.WinPanel)
        self.gridFrame_icp.setFrameShadow(QtWidgets.QFrame.Raised)
        self.gridFrame_icp.setObjectName("gridFrame_icp")
        self.gridLayout_3.setContentsMargins(10, 10, 10, 10)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.lbl_loss_sigma.setObjectName("lbl_loss_sigma")
        self.gridLayout_3.addWidget(self.lbl_loss_sigma, 1, 0, 1, 1)
        self.lbl_icp_threshold.setObjectName("lbl_icp_threshold")
        self.gridLayout_3.addWidget(self.lbl_icp_threshold, 0, 0, 1, 1)
        self.lbl_icp_iteration.setObjectName("lbl_icp_iteration")
        self.gridLayout_3.addWidget(self.lbl_icp_iteration, 2, 0, 1, 1)
        self.sbx_icp_iteration.setProperty("value", 10)
        self.sbx_icp_iteration.setObjectName("sbx_icp_iteration")
        self.gridLayout_3.addWidget(self.sbx_icp_iteration, 2, 1, 1, 1)
        self.sbx_icp_threshold.setSingleStep(0.001)
        self.sbx_icp_threshold.setDecimals(3)
        self.sbx_icp_threshold.setProperty("value", 0.02)
        self.sbx_icp_threshold.setObjectName("sbx_icp_threshold")
        self.gridLayout_3.addWidget(self.sbx_icp_threshold, 0, 1, 1, 1)
        self.sbx_loss_sigma.setSingleStep(0.01)
        self.sbx_loss_sigma.setProperty("value", 0.05)
        self.sbx_loss_sigma.setObjectName("sbx_loss_sigma")
        self.gridLayout_3.addWidget(self.sbx_loss_sigma, 1, 1, 1, 1)
        self.verticalLayout_3.addWidget(self.gridFrame_icp)
        self.verticalLayout.addLayout(self.verticalLayout_3)

    # Final section setup
    def setupFinalSection(self):
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)

        self.buttonBox.setOrientation(Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel | QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)

        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self):
        _translate = QtCore.QCoreApplication.translate
        self.setWindowTitle(_translate("Dialog", "Parameter Setting"))
        self.lbl_depth_acm_count.setText(_translate("Dialog", "■ Depth 누적 횟수"))
        self.lbl_outlier_remove_parameters.setText(_translate("Dialog", "■ Outlier 제거 파라미터"))
        self.cbx_iqr.setText(_translate("Dialog", "1. IQR (Inter Quatile Range) 기반 아웃라이어 제거"))
        self.lbl_iqr3.setText(_translate("Dialog", "IQR3"))
        self.lbl_iqr1.setText(_translate("Dialog", "IQR1"))
        self.cbx_gaussian.setText(_translate("Dialog", "2. 가우시안(Gaussian) 통계 기반 아웃라이어 제거"))
        self.lbl_gaussian_threshold.setText(_translate("Dialog", "Threshold"))
        self.cbx_sor_filter.setText(_translate("Dialog", "3. SOR Filter (Statistical Outlier Remove Filter)"))
        self.lbl_nb_neighbors.setText(
            _translate("Dialog", "nb_neighbors (Number of neighbors around the target point.)"))
        self.lbl_std_ratio.setText(_translate("Dialog", "std_ratio (Standard deviation ratio.)"))
        self.lbl_icp_parameters.setText(_translate("Dialog", "■ ICP 파라미터"))
        self.lbl_loss_sigma.setText(_translate("Dialog", "Loss sigma"))
        self.lbl_icp_threshold.setText(_translate("Dialog", "ICP Threshold"))
        self.lbl_icp_iteration.setText(_translate("Dialog", "ICP 반복 횟수"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    ui = ParameterDialog()
    ui.show()
    sys.exit(app.exec_())
