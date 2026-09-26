import os

try:
    from batoolset.pyqts.tools import getCtrlModifier
    from qtpy import QtGui, QtWidgets
    from qtpy.QtCore import Qt, QMimeData
    from qtpy.QtWidgets import QTableWidget, QApplication, QAbstractItemView

    class MyTableWidget(QTableWidget):
        def __init__(self, parent=None):
            super(MyTableWidget, self).__init__(parent)
            self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectItems)
            self.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
            self.copy = None

        def keyPressEvent(self, event):
            key = event.key()
            if key == Qt.Key_Delete:
                item = self.get_selected_item()
                if item is not None:
                    item.setText('')
            elif key == Qt.Key_C and (event.modifiers() & getCtrlModifier()):
                self.copy = self.get_selection()
            elif key == Qt.Key_V and (event.modifiers() & getCtrlModifier()):
                if self.copy is not None:
                    item = self.get_selected_item()
                    if item is not None:
                        item.setText(self.copy)
            else:
                super(MyTableWidget, self).keyPressEvent(event)

        def get_selection(self):
            selection = []
            items = self.selectedItems()
            for item in items:
                print('selected', item.text())
                selection.append(item.text())
            if len(selection) == 0:
                return None
            if len(selection) == 1:
                return selection[0]
            return selection

        def get_selected_item(self):
            items = self.selectedItems()
            for item in items:
                return item

except ImportError:
    pass


if __name__ == "__main__":
    pass
