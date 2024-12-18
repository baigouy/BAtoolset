from batoolset.settings.global_settings import set_UI # set the UI to qtpy
set_UI()
import os
import traceback
from timeit import default_timer as time
from qtpy.QtCore import QSize, QTimer,Qt
from natsort import natsorted
import glob
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import QPushButton, QWidget, QLineEdit, QApplication, QGridLayout, QStyle, QLabel, QTextEdit
from batoolset.dialogs.opensave import openFileNameDialog, openDirectoryDialog
import sys
import os



class QTextEditDND(QTextEdit):

    def __init__(self, title, parent, tip_text=None, objectName='', allow_multiple_drop=False):
        super().__init__(title, parent)
        self.allow_multiple_drop = allow_multiple_drop
        if tip_text is not None:
            self.setPlaceholderText(tip_text)
        self.setAcceptDrops(True)
        self.setObjectName(objectName)
        self.setAcceptRichText(True)

        # Make the QTextEdit single-line
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setLineWrapMode(QTextEdit.NoWrap)

        # Set a fixed height or minimum height to ensure the QTextEdit doesn't grow vertically
        self.setFixedHeight(30)  # or self.setMinimumHeight(20)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            text = []
            for txt in e.mimeData().urls():
                text.append(txt.toLocalFile())
            if not self.allow_multiple_drop or len(text) == 1:
                self.setPlainText(text[0])
            else:
                # self.setPlainText('\n'.join(text))
                self.setPlainText(str(text))


class QLineEditDND(QLineEdit):

    def __init__(self, title, parent, tip_text=None, objectName='',allow_multiple_drop=False):
        super().__init__(title, parent, objectName=objectName)
        self.allow_multiple_drop=allow_multiple_drop
        # self.setAcceptRichText(True)
        if tip_text is not None:
            self.setPlaceholderText(tip_text)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, e):
        # TODO test on osX to see if that works
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        if e.mimeData().hasUrls():
            text=[]
            for txt in e.mimeData().urls():
                text.append(txt.toLocalFile())
            if not self.allow_multiple_drop or len(text)==1:
                self.setText(text[0])
            else:
                self.setText(str(text))

        # TODO URGENT I guess I should activate the lines below
        #     e.accept()
        # else:
        #     e.ignore()

class OpenFileOrFolderWidget(QWidget):

    def __init__(self, parent_window=None, add_timer_to_changetext=False, show_ok_or_not_icon=False, label_text=None,
                 is_file=False, extensions="All Files (*);;", show_size=False, tip_text=None, objectName='',finalize_text_change_method_overrider=None, allow_multiple_drop=False, use_qtextedit=False):
        super().__init__(parent=parent_window)
        # print('changed method',finalize_text_change_method_overrider)
        self.finalize_text_change_method_overrider = finalize_text_change_method_overrider
        self.ok_ico = self.style().standardIcon(QStyle.SP_DialogYesButton).pixmap(QSize(12, 12))
        self.not_ok_ico = self.style().standardIcon(QStyle.SP_DialogNoButton).pixmap(QSize(12, 12))
        self.show_ok_or_not_icon = show_ok_or_not_icon
        self.parent_window = parent_window
        self.add_timer_to_changetext = add_timer_to_changetext
        self.label_text = label_text
        self.is_file = is_file
        self.tip_text = tip_text
        # self.is_file_or_folder = is_file_or_folder
        self.extensions = extensions
        self.show_size = show_size
        self.time_of_last_change = None
        self.allow_multiple_drop = allow_multiple_drop
        self.use_qtextedit = use_qtextedit
        self.initUI(objectName)

    def initUI(self, objectName):
        layout = QGridLayout()
        size_begin = 80
        size_label = 0
        size_end = 20
        size_warning = 0
        size_input_size = 0

        if self.label_text is not None:
            size_label = 10
            size_begin -= size_label

        if self.show_ok_or_not_icon:
            size_warning = 3
            size_begin -= size_warning

        if self.show_size:
            size_input_size = 5
            size_begin -= size_input_size

        layout.setColumnStretch(0, size_label)
        layout.setColumnStretch(1, size_begin)
        layout.setColumnStretch(2, size_end)
        layout.setColumnStretch(3, size_warning)
        layout.setColumnStretch(4, size_input_size)
        layout.setHorizontalSpacing(3)
        layout.setVerticalSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        if not self.use_qtextedit:
            self.path = QLineEditDND('', self, tip_text=self.tip_text, objectName=objectName, allow_multiple_drop=self.allow_multiple_drop)
        else:
            self.path = QTextEditDND('', self, tip_text=self.tip_text, objectName=objectName,                                 allow_multiple_drop=self.allow_multiple_drop)

        if self.add_timer_to_changetext:

            qtimer = QTimer()
            qtimer.setSingleShot(True)
            # self.path.textChanged.connect(lambda: qtimer.start(600))
            if isinstance(self.path, QLineEdit):
                self.path.textChanged.connect(lambda x: qtimer.start(600)) # somehow my weird bug is here and only in some conditions
            else:
                self.path.textChanged.connect(lambda: qtimer.start(600))
            # # Is there a better way TODO that ????
            # timer.timeout.connect(self._text_changed)
            qtimer.timeout.connect(self.finalize_text_change)
        if isinstance(self.path, QLineEdit):
            self.path.setDragEnabled(True)

        open_ico = QIcon.fromTheme("folder-open")
        open_button = QPushButton(open_ico, "Open", self)

        open_button.setFocusPolicy(Qt.NoFocus)
        # self.path.setDefault(True)
        # open_button.setDefault(False) # prevent it from
        # bt_width = open_button.fontMetrics().boundingRect(open_button.text()).width() + 30
        # open_button.setMaximumWidth(bt_width)
        # if self.is_file_or_folder:
        #     open_button.clicked.connect(self.open_file_or_folder)
        # el
        if self.is_file:
            open_button.clicked.connect(self.open_file)
        else:
            open_button.clicked.connect(self.open_folder)

        self.ok_or_not_ico = QLabel('')
        self.ok_or_not_ico.setPixmap(self.not_ok_ico)

        if self.show_ok_or_not_icon:
            layout.addWidget(self.ok_or_not_ico, 0, 3)

        if self.label_text is not None:
            layout.addWidget(QLabel(self.label_text), 0, 0)

        self.size_label = QLabel('0000')

        if self.show_size:
            layout.addWidget(self.size_label, 0, 4)

        layout.addWidget(self.path, 0, 1)
        layout.addWidget(open_button, 0, 2)
        self.setLayout(layout)

    def open_folder(self):
        if isinstance(self.path, QLineEdit):
            if os.path.isdir(self.path.text()):
                self.output_file_or_folder = openDirectoryDialog(parent_window=self.parent_window)
            else:
                self.output_file_or_folder = openDirectoryDialog(parent_window=self.parent_window,
                                                                                     path=self.path.text())
            if self.output_file_or_folder is not None:
                self.path.setText(self.output_file_or_folder)
        else:
            if os.path.isdir(self.path.toPlainText()):
                self.output_file_or_folder = openDirectoryDialog(parent_window=self.parent_window)
            else:
                self.output_file_or_folder = openDirectoryDialog(parent_window=self.parent_window,
                                                                 path=self.path.toPlainText())
            if self.output_file_or_folder is not None:
                self.path.setPlainText(self.output_file_or_folder)

    def open_file(self):
        # print('I am in there')
        #
        # try:
        #     print('in',self.path.text())
        # except:
        #     print('in',self.path.toPlainText())

        if isinstance(self.path, QLineEdit):
            if os.path.isfile(self.path.text()):
                self.output_file_or_folder = openFileNameDialog(parent_window=self.parent_window,
                                                                                    extensions=self.extensions)
            else:
                self.output_file_or_folder = openFileNameDialog(parent_window=self.parent_window,
                                                                                    extensions=self.extensions,
                                                                                    path=self.path.text())
            if self.output_file_or_folder:
                self.path.setText(self.output_file_or_folder)
        else:
            if os.path.isfile(self.path.toPlainText()):
                self.output_file_or_folder = openFileNameDialog(parent_window=self.parent_window,
                                                                extensions=self.extensions)
            else:
                self.output_file_or_folder = openFileNameDialog(parent_window=self.parent_window,
                                                                extensions=self.extensions,
                                                                path=self.path.toPlainText())

            if self.output_file_or_folder:
                self.path.setPlainText(self.output_file_or_folder)

        # try:
        #     print('out',self.path.text())
        # except:
        #     print('out',self.path.toPlainText())

    def set_time_of_last_change(self):
        self.time_of_last_change = time()
        # print('I have been changed', self.time_of_last_change, self) # to debug

    def get_time_of_last_change(self):
        return  self.time_of_last_change

    # def _text_changed(self):
    #     # this stores the time of last change --> can easily determine when the object was last changed
    #     self.set_time_of_last_change()
    #     # self.finalize_text_change()

    def text(self):
        if isinstance(self.path, QLineEdit):
            if self.path.text().strip() == '':
                return None
            return self.path.text()
        else:
            if self.path.toPlainText().strip() == '':
                return None
            return self.path.toPlainText()

    def finalize_text_change(self):
            # print('finalize_text_change_method_overrider',self.finalize_text_change_method_overrider)
        # try:
            self.set_time_of_last_change()
            if self.finalize_text_change_method_overrider is not None:
                # print('calling neo finalizer')
                # if an overriding method was defined --> use it otherwise use default
                self.finalize_text_change_method_overrider()
            # else:
            #     #
            #     print('in', self.path.text())
            #     print('in', self.path.toPlainText())
                # pass
        # except:
        #     # traceback.print_stack()
        #     traceback.print_exc()
        #     pass
        # pass

    def set_icon_ok(self, ok):
        if not self.show_ok_or_not_icon:
            return
        if ok:
            self.ok_or_not_ico.setPixmap(self.ok_ico)
        else:
            self.ok_or_not_ico.setPixmap(self.not_ok_ico)

    def set_size(self, size):
        self.size_label.setText(size)

    def get_list_using_glob(self):
        try:
            if isinstance(self.path, QLineEdit):
                filenames = [file for file in glob.glob(self.path.text())]
            else:
                filenames = [file for file in glob.glob(self.path.toPlainText())]
            filenames = natsorted(filenames)  # human-like sorting of file names
            return filenames
        except:
            return None

if __name__ == '__main__':
    # just for a test
    app = QApplication(sys.argv)

    def new_method():
        print('tutu')


    use_qtextedit = True
    is_file=True
    ex = OpenFileOrFolderWidget(parent_window=None, add_timer_to_changetext=True, finalize_text_change_method_overrider=new_method, allow_multiple_drop=True, use_qtextedit=use_qtextedit, is_file=is_file)
    text = "This is <span style=\"color:green\">green text</span> and <span style=\"color:red\">red text</span>."
    try:
        ex.path.setText(text)
    except:
        ex.path.setHtml(text)
    ex.show()
    app.exec_()
