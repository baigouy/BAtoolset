import os
from batoolset.settings.global_settings import set_UI # set the UI to qtpy
set_UI()
from qtpy.QtCore import Qt
from qtpy.QtGui import QPalette, QPainter, QBrush, QColor, QPen, QFontMetrics
from qtpy.QtWidgets import QWidget


class Overlay(QWidget):

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        palette = QPalette(self.palette())
        # palette.setColor(palette.Background, Qt.transparent) # change due to a shift to the qtpy paradigm
        self.setPalette(palette)

    def paintEvent(self, event):
        painter = QPainter()
        try:
            painter.begin(self)
            painter.save()
            painter.setRenderHint(QPainter.Antialiasing)
            painter.fillRect(event.rect(), QBrush(QColor(255, 255, 255, 127)))
            painter.setPen(QPen(Qt.NoPen))

            painter.setPen(QColor('red'))
            f = painter.font()
            f.setBold(True)
            f.setPointSize(20)

            message = 'Please wait...'
            fm = QFontMetrics(f)
            txt_width = fm.width(message)
            txt_height = fm.height()

            painter.setFont(f)

            flags = Qt.AlignTop | Qt.AlignLeft | Qt.TextSingleLine
            painter.drawText(int(self.width() / 2 - txt_width), int(self.height() / 2 - self.fontMetrics().height()), txt_width,
                             txt_height, flags, message)

            painter.restore()
        finally:

            painter.end()
