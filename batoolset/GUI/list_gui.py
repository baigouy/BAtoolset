# support list for writing or not stuff
import os
from batoolset.settings.global_settings import set_UI # set the UI to qtpy
set_UI()
from qtpy import QtCore, QtWidgets
from qtpy.QtCore import Qt,Signal
from qtpy.QtWidgets import QWidget, QAbstractItemView, QListWidget, QVBoxLayout, QListWidgetItem, QToolBar, \
    QToolButton, \
    QLabel, QAction
import pyperclip as clip
from batoolset.dialogs.opensave import saveFileDialog
from batoolset.utils.loadlist import loadlist, save_list_to_file
import os
import platform
import subprocess
from batoolset.files.tools import smart_name_parser
import qtawesome as qta
from natsort import natsorted
from batoolset.tools.logger import TA_logger
logger = TA_logger()


__DEBUG__ = True

IMAGES_2D = ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.tga', '.bmp']
STACKS = ['.tif', '.tiff', '.lif', '.czi', '.lif', '.lsm']
LISTS = ['.lst', '.txt']
FASTA = ['.fa','.fna','.fasta']
PYTA_DEFAULT = IMAGES_2D + LISTS

class CustomListWidget(QListWidget):
    orderChanged = Signal()

    def dropEvent(self, event):
        super().dropEvent(event)
        self.orderChanged.emit()

class ListGUI(QWidget):
    orderChanged = Signal()
    deletePressed = Signal() # can be used in parent to detect that this was called allowing some action to be taken (more elegant than the callback or the method overlaod in several aspects)!
    # TODO define supported formats --> TODO
    def __init__(self, parent=None, file_to_load=None, supported_files=None, custom_label='Please Drag and Drop files below:', default_double_click_behaviour='full_no_ext', enable_folder_DND=True, detect_list_order_change=False):
        super().__init__(parent)
        if detect_list_order_change:
            self.list = CustomListWidget(self)  # a list that contains files to read or play with
        #     self.list.orderChanged.connect(self.check_order_changed)
        else:
            self.list = QListWidget(self)
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        # self.list.selectionModel().selectionChanged.connect(self.selectionChanged)  # connect it to sel change

        # self.list.selectionModel().selectionChanged.connect(self.listSelectionChanged)

        self.list.itemDoubleClicked.connect(self.open_corresponding_folder)

        self.enable_folder_DND = enable_folder_DND
        self.default_double_click_behaviour = default_double_click_behaviour

        layout = QVBoxLayout()
        lab = QLabel(custom_label)
        layout.addWidget(lab)
        layout.addWidget(self.list)

        self.setLayout(layout)
        # allow support of DND
        self.setAcceptDrops(True)
        self.supported_files = supported_files

        # status_bar = QStatusBar()
        # layout.addWidget(status_bar)
        # self.list_commands(**kwargs)
        self.list_commands()

        if file_to_load is not None:
            self.load_list(file_to_load)
            # if __DEBUG__:
            # fake sel list
            # pass

        # allow move objects within the list
        self.list.setDragDropMode(QAbstractItemView.InternalMove)  # j'adore --> maybe put this as an option

        # allows sorting of lists
        # self.list.setSortingEnabled(True) # we don't want list to be sorted by default
        # self.list.sortItems()  # ascending by default
        self.list.sortItems(QtCore.Qt.DescendingOrder)  # todo MAKE it a sort button

        # both lines below support objects handling directly from the list --> a gain of time
        self.list.itemChanged.connect(self.item_changed)
        self._object_dict = {} # dirty hack to store objects and allow for DND of objects

    def item_changed(self, item):
        obj = self._object_dict.get(str(item))  # Get the object from the dictionary
        if obj is not None:
            # Do something with the object
            print("Object changed:", obj)
    def check_order_changed(self):
        # Implement your order changed logic here
        print("Order changed!")
        # self.setWindowTitle('test')

    def open_corresponding_folder(self, double_clicked_item):
        try:
            self.open_folder(smart_name_parser(double_clicked_item.toolTip(), ordered_output=[self.default_double_click_behaviour])[0]) #'full_no_ext'
        except:
            # no big deal if it fails
            pass

    def open_folder(self, path):
        if platform.system() == 'Darwin':
            subprocess.Popen(['open', path])
        elif platform.system() == 'Windows':
            os.startfile(path)
        elif platform.system() == 'Linux':
            subprocess.Popen(["xdg-open", path])

    def list_commands(self): #,**kwargs):

        # buttons_to_hide = []
        # if 'buttons_to_remove' in kwargs:
        #     buttons_to_hide = kwargs['buttons_to_remove']
        #     if isinstance(buttons_to_hide, str):
        #         buttons_to_hide = [buttons_to_hide]

        # self.penSize = QSpinBox(objectName='penSize')
        # self.penSize.setSingleStep(1)
        # self.penSize.setRange(1, 256)
        # self.penSize.setValue(3)
        # self.penSize.valueChanged.connect(self.penSizechange)

        # self.channels = QComboBox(objectName='channels')
        # self.channels.addItem("merge")
        # self.channels.addItem("0")
        # self.channels.addItems(["1", "2", "3"])
        # self.channels.currentIndexChanged.connect(self.channelChange)

        self.tb = QToolBar()
        #
        toolButton = QToolButton()
        # toolButton.setText('Draw')


        save_action = QAction(qta.icon('fa5.save'), 'Save image list', self)
        save_action.triggered.connect(self.save_list)
        # END KEEP
        toolButton.setDefaultAction(save_action)
        self.tb.addWidget(toolButton)
        #
        # toolButton2 = QToolButton()
        # toolButton2.setText('Draw')
        # toolButton2.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        # tb.addWidget(toolButton2)
        #
        # toolButton3 = QToolButton()
        # toolButton3.setText('Draw')
        # toolButton3.setIcon(self.style().standardIcon(QStyle.SP_DialogCancelButton))
        # tb.addWidget(toolButton3)
        #
        # toolButton4 = QToolButton()
        # toolButton4.setText('Draw')
        # toolButton4.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        # tb.addWidget(toolButton4)

        toolButton5 = QToolButton()
        # toolButton5.setText("Delete")
        # toolButton5.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        # toolButton5.addAction(dqsdqsdq)
        # pkoi marche pas
        # KEEP

        delete_sel_action = QAction(qta.icon('mdi.delete-outline'), 'Delete selection', self)
        # toolButton5.addAction(delete_sel_action) # this stuff adds a dropwdon that can be opened by pressing a long time on the button --> keep in mind cause can be useful
        delete_sel_action.triggered.connect(self.removeSel)
        # END KEEP
        toolButton5.setDefaultAction(delete_sel_action)
        self.tb.addWidget(toolButton5)

        toolButton6 = QToolButton()
        # toolButton6.setText("Copy")
        # toolButton6.setIcon(qta.icon('fa5.copy'))
        copy_action = QAction(qta.icon('fa5.copy'), 'Copy selection to the system clipboard', self)
        copy_action.triggered.connect(self.sel_to_clipboard)
        toolButton6.setDefaultAction(copy_action)
        # toolButton6.ad#dAction(print('toto'))
        self.tb.addWidget(toolButton6)

        toolButton7 = QToolButton()
        # toolButton7.setText("Paste")
        # toolButton7.setIcon(qta.icon('fa.paste'))
        paste_action = QAction(qta.icon('fa.paste'), 'Paste the system clipboard to the list', self)
        paste_action.triggered.connect(self.paste_from_clipboard)
        toolButton7.setDefaultAction(paste_action)
        self.tb.addWidget(toolButton7)

        toolButton8 = QToolButton()
        sort_action = QAction(qta.icon('mdi.sort-alphabetical-ascending'), 'Sort list (using the Natsort algorithm)', self)
        sort_action.triggered.connect(self.natsort_list)
        toolButton8.setDefaultAction(sort_action)
        self.tb.addWidget(toolButton8)

        # tb.addWidget(self.channels)

        # tb.addAction("Save")
        # tb.addAction("sq...")
        # tb.addWidget(self.penSize)

        # could have show and hide mask here --> see how I can do that --> but do not delete it in fact

        self.layout().addWidget(self.tb)

    # disable item internal move and list addition, just selection is kept
    def freeze(self, bool):
        self.tb.setEnabled(not bool)
        # TODO disable DND of list component
        # if bool:
        # self.list.setDragDropMode(QAbstractItemView.InternalMove)
        # self.setAcceptDrops(True)
        # else:
        # self.list.setDragDropMode(QAbstractItemView.InternalMove)
        self.setAcceptDrops(not bool)
        # how to disable item reorder
        # QGraphicsItem.ItemIsMovable
        # we also prevent internal drag --> it's a way to keep the list functional



        # ça ne marche pas
        # print(bool)
        if bool:

            # print('entering o,ne')

            # disable item internal move
            for iii in range(self.list.count()):
                item = self.list.item(iii)

                # is there a bug here
                if  item.flags() & Qt.ItemIsDragEnabled:
                    item.setFlags(item.flags() ^ Qt.ItemIsDragEnabled)
                # item.setFlags(item.flags() ^ Qt.ItemIsDropEnabled)

                # if iii == 0:
                #     if item.flags() & Qt.ItemIsDragEnabled:
                #         print( "ItemIsDragEnabled", item)
                #     else:
                #         print("not ItemIsDragEnabled", item)

        else:
            # allow item internal move
            # print('entering two')
            for iii in range(self.list.count()):
                item = self.list.item(iii)
                item.setFlags(item.flags() | Qt.ItemIsDragEnabled)
                # item.setFlags(item.flags() | Qt.ItemIsDropEnabled)
                # if iii == 0:
                #     if item.flags() & Qt.ItemIsDragEnabled:
                #         print("ItemIsDragEnabled", item)
                #     else:
                #         print("not ItemIsDragEnabled", item)


    def load_list(self, file_to_load):
        files = loadlist(file_to_load)
        if files is not None:
            # populate the list
            for file in files:
                self.add_file_to_list_if_supported(file)
        else:
            print('list could not be loaded/invalid list --> sorry')

    # def selectionChanged(self):
    #     print('selection changed to override')

    # allow DND
    def dragEnterEvent(self, event):
        if not self.acceptDrops():
            logger.warning('Drag and drop not supported in "Preview" mode, please select another tab.')
            event.ignore()
            return
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if not self.acceptDrops():
            logger.warning('Drag and drop not supported in "Preview" mode, please select another tab.')
            event.ignore()
            return
        if event.mimeData().hasUrls():
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    # handle DND on drop
    def dropEvent(self, event):
        super().dropEvent(event)
        self.orderChanged.emit()

        if not self.acceptDrops():
            logger.warning('Drag and drop not supported in "Preview" mode, please select another tab.')
            event.ignore()
            return

        if event.mimeData().hasUrls():
            event.setDropAction(QtCore.Qt.CopyAction)
            event.accept()
            urls = []
            for url in event.mimeData().urls():
                urls.append(url.toLocalFile())
            # add all dropped items to the list
            for url in urls:
                # item = QListWidgetItem(os.path.basename(url), self.list)
                # item.setToolTip(url)
                # self.list.addItem(item)
                self.add_file_to_list_if_supported(url)
        else:
            event.ignore()

    def is_file_supported(self, file):
        if self.supported_files is None:
            # accept all by default
            return True
        if smart_name_parser(file, ordered_output=['ext'])[0].lower() in self.supported_files:
            return True
        return False

    def add_file_to_list_no_check(self, file):
        item = QListWidgetItem(os.path.basename(file), self.list)
        item.setToolTip(file)
        self.list.addItem(item)

    def add_object_to_list_no_check(self, obj): #added this to add non file objects to the list
        item = QListWidgetItem(str(obj), self.list)
        item.setToolTip(str(obj))
        # item.setData(Qt.UserRole, obj)
        # item.setData(Qt.UserRole, str(obj))
        self._object_dict[str(item)] = obj  # Store a reference to the object in the dictionary
        self.list.addItem(item)

    def add_to_list(self, files, check_if_supported=True):
        if isinstance(files, str):
            if check_if_supported:
                self.add_file_to_list_if_supported(files)
            else:
                self.add_file_to_list_no_check(files)
        elif isinstance(files, list):
            for file in files:
                if check_if_supported:
                    self.add_file_to_list_if_supported(file)
                else:
                    self.add_file_to_list_no_check(file)
        else:
            print('Non supported, cannot be added to list', files)

    def list_files_in_directory(self, directory_path):
        files = [os.path.join(directory_path,f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        # print(files)
        return files

    def add_file_to_list_if_supported(self, file):
        # print('checking if supported', file)
        if self.is_file_supported(file) and os.path.isfile(file):
            ext = smart_name_parser(file, ordered_output=['ext'])[0].lower()
            if ext in LISTS:
                TA_list_of_files = loadlist(file)
                return self.add_to_list(TA_list_of_files, check_if_supported=True)
            else:
                self.add_file_to_list_no_check(file)
        elif self.enable_folder_DND and os.path.isdir(file):
            files = self.list_files_in_directory(file)
            for file in files:
                self.add_file_to_list_if_supported(file)


    def get_full_list(self):
        # return the complete list of files
        selected_items = []
        for i in range(self.list.count()):
            if str(self.list.item(i)) in self._object_dict:
                selected_items.append(self._object_dict[str(self.list.item(i))])
            else:
                selected_items.append(self.list.item(i).toolTip())
        return selected_items

    def get_selection(self, mode='single'):
        selected_items = self.list.selectedItems()
        if selected_items:
            if mode == 'single':
                if str(selected_items[0]) in self._object_dict:
                    return self._object_dict[str(selected_items[0])]
                # if selected_items[0].data(Qt.UserRole):
                #     return selected_items[0].data(Qt.UserRole)
                return selected_items[0].toolTip()
            else:
                sel = []
                for selec in selected_items:
                    if str(selec) in self._object_dict:
                        sel.append(self._object_dict[str(selec)])
                    # elif selec.data(Qt.UserRole):
                    #     sel.append(selec.data(Qt.UserRole))
                    else:
                        sel.append(selec.toolTip())
                return sel
        else:
            return None

    def get_selection_index(self, mode='single'):
        try:
            # selected_items = self.list.selectedIndexes() # this is a weird model index object that is largely useless
            selected_items = [s.row() for s in self.list.selectedIndexes()]
            # print('selected_items',selected_items)
            if selected_items:
                if mode == 'single':
                    # print('sel', selected_items[0])
                    return selected_items[0]
                else:
                    # sel = []
                    # for selec in selected_items:
                    #     sel.append(selec.toolTip())
                    return selected_items
            # else:
        except:
            pass
        return None

    def save_list(self):
        # on va sauver la liste ici
        lst = self.get_full_list()
        if lst:
            path = smart_name_parser(lst[0], ordered_output='parent')
            path = os.path.join(path, 'list.lst')

            output_file = saveFileDialog(parent_window=self, path=path,
                                         extensions="Lists (*.lst);;Supported Files (*.lst *.txt);;All Files (*)",
                                         default_ext='.lst')
            if output_file is not None:
                # now do save the list in the folder of the parent of the first file
                save_list_to_file(lst, output_file)
                # print(lst, output_file)

    def removeSel(self):
        selected_items = self.list.selectedItems()
        self.list.clearSelection()  # Clear selection # required to prevent selection from changing and the software to want to update display or anything alike
        if not selected_items:
            self.clearList()
            self.deletePressed.emit() # send the delete action to the parent
            return

        # Remove selected items from the list
        for item in selected_items:
            # Remove the item from the dictionary
            if str(item) in self._object_dict:
                del self._object_dict[str(item)]
        for item in selected_items:
            # print('looping')
            self.list.takeItem(self.list.row(item))
        self.deletePressed.emit() # send the delete action to the parent

    # faire une copie dans le clip de la selection
    def sel_to_clipboard(self):
        sel = self.get_selection(mode='all')
        if sel is not None:
            clip.copy('\n'.join(sel))

    def paste_from_clipboard(self):
        files = clip.paste()
        # try add this to the list
        try:
            # print(files)
            splitted_files = files.split('\n')
            # print(splitted_files)
            self.add_to_list(splitted_files, check_if_supported=True)
        except:
            # no big deal if fails
            pass

    # TODO maybe add sort by time as there is even an icon for that and can be useful for my leica files !!!
    # TODO copy icons if they exist on natsort ok for now it's really a detail...
    def natsort_list(self):
        full_list = self.get_full_list()
        if full_list is not None:
            self.clearList()
            self.add_to_list(natsorted(full_list), check_if_supported=False)

    def clearList(self):
        self.list.clear()
        del self._object_dict
        self._object_dict = {}


if __name__ == '__main__':
    # TODO add a main method so it can be called directly
    # maybe just show a canvas and give it interesting props --> TODO --> really need fix that too!!!
    import sys
    from qtpy.QtWidgets import QApplication

    # should probably have his own scroll bar embedded somewhere

    app = QApplication(sys.argv)

    w = ListGUI(file_to_load='/E/Sample_images/sample_images_PA/mini/list.lst')

    # w.freeze(False)
    # w.freeze(True)

    w.show()

    print('get_full_list', w.get_full_list())  # very good and useful
    print('sel', w.get_selection(mode='all'))

    sys.exit(app.exec_())
