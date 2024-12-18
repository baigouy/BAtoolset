from batoolset.settings.global_settings import set_UI # set the UI to qtpy
set_UI()
import sys
from qtpy.QtCore import QRect, QTimer
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QWidget

from batoolset.drawings.widgets.vectorial import VectorialDrawPane
from qtpy.QtWidgets import QMenu, QApplication
from qtpy import QtCore, QtGui

from batoolset.img import toQimage
from batoolset.tools.logger import TA_logger # logging

logger = TA_logger()

class Createpaintwidget(QWidget):

    def __init__(self):
        super().__init__()
        self.vdp = VectorialDrawPane(active=False) #, demo=True
        self.image = None
        self.imageDraw = None
        self.cursor = None
        self.maskVisible = True
        self.scale = 1.0
        self.drawing = False
        self.brushSize = 3
        self._clear_size = 30
        self.drawColor = QtGui.QColor(QtCore.Qt.red) # blue green cyan
        self.eraseColor = QtGui.QColor(QtCore.Qt.black)
        self.cursorColor = QtGui.QColor(QtCore.Qt.green)
        self.lastPoint = QtCore.QPointF()
        self.change = False
        # KEEP IMPORTANT required to track mouse even when not clicked
        self.setMouseTracking(True)  # KEEP IMPORTANT
        self.scrollArea = None
        self.statusBar = None

    def setImage(self, img):
        if img is None:
            self.image = None
            self.imageDraw = None
            self.update()
            return
        else:
            self.image = toQimage(img) #.getQimage() # bug is here

            # self.image = QPixmap(100,200).toImage()
        width = self.image.size().width()
        height = self.image.size().height()
        top = self.geometry().x()
        left = self.geometry().y()
        self.setGeometry(top, left, int(width*self.scale), int(height*self.scale))
        self.imageDraw = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
        self.imageDraw.fill(QtCore.Qt.transparent)
        self.cursor = QtGui.QImage(self.image.size(), QtGui.QImage.Format_ARGB32)
        self.cursor.fill(QtCore.Qt.transparent)
        self.update()

    def mousePressEvent(self, event):
        if not self.hasMouseTracking():
            return
        self.clickCount = 1
        if self.vdp.active:
            self.vdp.mousePressEvent(event)
            self.update()
            return

        if event.buttons() == QtCore.Qt.LeftButton or event.buttons() == QtCore.Qt.RightButton:
            self.drawing = True
            zoom_corrected_pos = event.position() / self.scale
            self.lastPoint = zoom_corrected_pos
            self.drawOnImage(event)

    def mouseMoveEvent(self, event):
        if not self.hasMouseTracking():
            return
        # print('in mouse move', self.hasMouseTracking(), self.drawing, self.vdp.active)
        if self.statusBar:
            zoom_corrected_pos = event.position() / self.scale
            self.statusBar.showMessage('x=' + str(zoom_corrected_pos.x()) + ' y=' + str(
                zoom_corrected_pos.y()))
        if self.vdp.active:
            self.vdp.mouseMoveEvent(event)
            region = self.scrollArea.widget().visibleRegion()
            self.update(region)
            return
        self.drawOnImage(event)

    def drawOnImage(self, event):
        zoom_corrected_pos = event.position() / self.scale
        if self.drawing and (event.buttons() == QtCore.Qt.LeftButton or event.buttons() == QtCore.Qt.RightButton):
            # now drawing or erasing over the image
            painter = QtGui.QPainter(self.imageDraw)
            if event.buttons() == QtCore.Qt.LeftButton:
                painter.setPen(QtGui.QPen(self.drawColor, self.brushSize, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                          QtCore.Qt.RoundJoin))
            else:
                painter.setPen(QtGui.QPen(self.eraseColor, self.brushSize, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                          QtCore.Qt.RoundJoin))
            if self.lastPoint != zoom_corrected_pos:
                painter.drawLine(self.lastPoint, zoom_corrected_pos)
            else:
                # if zero length line then draw point instead
                painter.drawPoint(zoom_corrected_pos)
            painter.end()

        # Drawing the cursor TODO add boolean to ask if drawing cursor should be shown
        painter = QtGui.QPainter(self.cursor)
        # We erase previous pointer
        r = QtCore.QRect(QtCore.QPoint(), self._clear_size * QtCore.QSize() * self.brushSize)
        painter.save()
        r.moveCenter(self.lastPoint.toPoint())
        painter.setCompositionMode(QtGui.QPainter.CompositionMode_Clear)
        painter.eraseRect(r)
        painter.restore()
        # draw the new one
        painter.setPen(QtGui.QPen(self.cursorColor, 2, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                  QtCore.Qt.RoundJoin))
        painter.drawEllipse(zoom_corrected_pos, int(self.brushSize / 2.),
                            int(self.brushSize / 2.))
        painter.end()
        region = self.scrollArea.widget().visibleRegion()
        self.update(region)

        # required to erase mouse pointer
        self.lastPoint = zoom_corrected_pos

    def mouseReleaseEvent(self, event):
        if not self.hasMouseTracking():
            return
        if self.vdp.active:
            self.vdp.mouseReleaseEvent(event)
            self.update()  # required to update drawing
            return
        if event.button == QtCore.Qt.LeftButton:
            self.drawing = False
        if self.clickCount == 1:
            QTimer.singleShot(QApplication.instance().doubleClickInterval(),
                              self.updateButtonCount)

    # adds context/right click menu but only in vectorial mode
    def contextMenuEvent(self, event):
        if not self.vdp.active:
            return
        cmenu = QMenu(self)
        newAct = cmenu.addAction("New")
        opnAct = cmenu.addAction("Open")
        quitAct = cmenu.addAction("Quit")
        action = cmenu.exec_(self.mapToGlobal(event.position()))
        if action == quitAct:
            sys.exit(0)

    def updateButtonCount(self):
        self.clickCount = 1

    def mouseDoubleClickEvent(self, event):
        self.clickCount = 2
        self.vdp.mouseDoubleClickEvent(event)

    def paintEvent(self, event):
        canvasPainter = QtGui.QPainter(self)
        # the scrollpane visible region
        visibleRegion = self.scrollArea.widget().visibleRegion()
        # the corresponding rect
        visibleRect = visibleRegion.boundingRect()
        # the visibleRect taking zoom into account
        scaledVisibleRect = QRect(int(visibleRect.x() / self.scale), int(visibleRect.y() / self.scale),
                                  int(visibleRect.width() / self.scale), int(visibleRect.height() / self.scale))
        if self.image is None:
            canvasPainter.eraseRect(visibleRect)
            canvasPainter.end()
            return

        canvasPainter.drawImage(visibleRect, self.image, scaledVisibleRect)
        if not self.vdp.active and self.maskVisible:
            canvasPainter.drawImage(visibleRect, self.imageDraw, scaledVisibleRect)
            # should draw the cursor
        canvasPainter.drawImage(visibleRect, self.cursor, scaledVisibleRect)

        if self.vdp.active:
            self.vdp.paintEvent(canvasPainter, scaledVisibleRect)
        canvasPainter.end()
