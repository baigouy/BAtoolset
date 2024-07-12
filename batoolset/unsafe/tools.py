from batoolset.settings.global_settings import set_UI # set the UI to qtpy
set_UI()
import sys
from qtpy.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QTextEdit, QLabel, QMessageBox
from batoolset.img import Img

def execute_code(code, parent=None):
    # code = self.code_editor.toPlainText()

    # Display a warning dialog before executing the code
    warning_dialog = QMessageBox(parent=parent)
    warning_dialog.setWindowTitle("Warning")
    warning_dialog.setText("Executing arbitrary code can be dangerous.\nAre you sure you want to proceed?")
    warning_dialog.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
    warning_dialog.setDefaultButton(QMessageBox.No)
    warning_dialog.setIcon(QMessageBox.Warning)

    # Execute the code if the user clicks "Yes"
    if warning_dialog.exec_() == QMessageBox.Yes:
        param = 10
        text = 'this is a test of your system'
        # print('before', locals())

        # local_namespace = {"param": param, "text": text, "img": img}
        local_namespace = {}

        # current_locals = locals()
        try:
            exec(code, locals(), local_namespace)
            # exec(code, current_locals, local_namespace)
            # img = local_namespace["img"]

            """
            example of calling the stuff

            # img=img[...,0]
            # text='tutu'
            # param+=6
            self.img=self.img[...,0]
            text='tutu'
            param+=6
            """

            # only allow very few number of variables to be edited -−> TODO
            if False:
                if False:  # below it gets all the variables that may cause trouble
                    # Update the local variables from the local namespace from the code (there are advantages to that but it may also cause problems)
                    for var_name, var_value in local_namespace.items():
                        if var_name in locals() and locals()[var_name] != var_value:
                            locals()[var_name] = var_value
                else:
                    # a slightly better version that excludes local variables with _
                    for var_name, var_value in local_namespace.items():
                        if not var_name.startswith("_") and var_name in locals() and locals()[var_name] != var_value:
                            locals()[var_name] = var_value
                            # globals()[var_name] = var_value

            # print('after', locals())
            #
            # if 'text' in local_namespace:
            #     text = local_namespace['text']
            # if 'param' in local_namespace:
            #     param = local_namespace['param']
            # if 'self.img' in local_namespace:  # that works also with self --> do this specifically for the image stuff
            #     self.img = local_namespace['self.img']
            #
            # # print("Result:", img.shape)  # Print the result in console for demonstration
            # print("Result: text", text)  # Print the result in console for demonstration
            # print("Result: param", param)
            # print("Result: self.img", self.img.shape)
            # print("Result:", param)

            # print("Result:", img.shape)  # Print the result in console for demonstration
        except Exception as e:
            print("Error:", e)


class CodeExecutorWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Code Executor")
        self.setGeometry(100, 100, 600, 400)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        self.code_editor = QTextEdit()
        layout.addWidget(self.code_editor)

        execute_button = QPushButton("Execute")
        execute_button.clicked.connect(lambda: execute_code(self.code_editor.toPlainText(), self))
        layout.addWidget(execute_button)

        self.warning_label = QLabel("Warning: Executing arbitrary code can be dangerous. Only rely on trusted sources.")
        self.warning_label.setStyleSheet("color: red")
        layout.addWidget(self.warning_label)

        self.img = Img('/E/Sample_images/sample_images_PA/mini_vide/focused_Series012.png') # that works and it is really cool
        # shall I take a user edited result instead and keep the original --> in a way yes be cause this is the only way to preseve the integrity of the images --> think of that --> I really need to think
        # but then I cannot share the images between all objects --> anyway this is not smart if they can be edited -->








if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CodeExecutorWindow()
    window.show()
    sys.exit(app.exec_())
