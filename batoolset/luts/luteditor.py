from batoolset.settings.global_settings import set_UI # set the UI to qtpy
set_UI()
import sys
from qtpy.QtWidgets import QApplication, QWidget, QGridLayout, QPushButton, QColorDialog, QLabel, QVBoxLayout
from qtpy.QtGui import QColor, QPixmap, QImage
from qtpy.QtCore import Qt
from batoolset.pyqts.tools import updateLutLabelPreview

class PaletteEditor(QWidget):
    def __init__(self, lut=None, parent=None):
        super().__init__(parent)

        self.button_width = 40  # Example button width
        self.button_height = 20  # Example button height
        self.spacing = 2  # Example spacing

        self.rows = 16
        self.columns = 16
        self.buttons = {}

        self.initUI()

        if lut:
            self.setLUT(lut)
        else:
            self.setDefaultLUT()

        # self.updatePreview()
        total_length = (self.button_width * self.columns) + ( self.spacing * (self.columns - 1))
        updateLutLabelPreview(self.getLUT(), self.previewLabel,total_length)

    def initUI(self):
        mainLayout = QVBoxLayout()

        gridLayout = QGridLayout()
        gridLayout.setSpacing(2)

        for row in range(self.rows):
            for col in range(self.columns):
                button = QPushButton()
                button.setFixedSize(self.button_width, self.button_height)
                button.setStyleSheet("background-color: white;")
                button.clicked.connect(lambda _, r=row, c=col: self.openColorDialog(r, c))

                gridLayout.addWidget(button, row, col)
                self.buttons[(row, col)] = button

        self.previewLabel = QLabel()
        gridLayout.addWidget(self.previewLabel, self.rows, 0, 1, self.columns)

        mainLayout.addLayout(gridLayout)
        self.setLayout(mainLayout)
        self.setWindowTitle('Palette Editor (256 entries)')
        self.show()

    def openColorDialog(self, row, col):
        button = self.buttons[(row, col)]
        current_color = button.palette().color(button.backgroundRole())

        color = QColorDialog.getColor(current_color, self)

        if color.isValid():
            button.setStyleSheet(f"background-color: {color.name()};")
            # self.updatePreview()
            total_length = (self.button_width * self.columns) + (self.spacing * (self.columns - 1))
            updateLutLabelPreview(self.getLUT(), self.previewLabel, total_length)

    def getLUT(self):
        lut = []
        for row in range(self.rows):
            for col in range(self.columns):
                button = self.buttons[(row, col)]
                color = button.palette().color(button.backgroundRole())
                lut.append(color.name())
        return lut

    def setLUT(self, lut):
        if len(lut) != 256:
            raise ValueError("LUT must contain exactly 256 entries")

        index = 0
        for row in range(self.rows):
            for col in range(self.columns):
                color = QColor(lut[index])
                if color.isValid():
                    self.buttons[(row, col)].setStyleSheet(f"background-color: {color.name()};")
                index += 1

    def setDefaultLUT(self):
        default_lut = [f"#{i:02x}{i:02x}{i:02x}" for i in range(256)]
        self.setLUT(default_lut)

    # def updatePreview(self):
    #     lut = self.getLUT()
    #     image = QImage(256, 16, QImage.Format_RGB32)
    #
    #     for i in range(256):
    #         color = QColor(lut[i])
    #         for j in range(16):
    #             image.setPixelColor(i, j, color)
    #
    #     pixmap = QPixmap.fromImage(image)
    #     total_width = (40 * self.columns) + (2 * (self.columns - 1))
    #     scaled_pixmap = pixmap.scaled(total_width, 16, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
    #     self.previewLabel.setPixmap(scaled_pixmap)


if __name__ == '__main__':
    # this is the first version of my LUT editor

    app = QApplication(sys.argv)

    # Create PaletteEditor with the default grayscale LUT
    ex = PaletteEditor()

    sys.exit(app.exec_())
