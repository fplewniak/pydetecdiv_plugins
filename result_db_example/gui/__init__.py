"""
GUI for an example plugin showing how to create, populate and query a database table to store results generated by a
plugin
"""
from PySide6.QtCore import QStringListModel, Qt
from PySide6.QtWidgets import QFrame, QFormLayout, QLabel, QComboBox, QListView, QDialogButtonBox, QDockWidget

from pydetecdiv.utils import singleton


@singleton
class DockWindow(QDockWidget):
    """
    A DockWidget to host the GUI for Example plugin's
    This is a singleton to avoid creating more than one window, but this is not compulsory and there may be several
    instance of such a window for a single plugin if needed.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle('Results in DB example plugin')
        self.setObjectName('Results in DB example plugin')

        self.form = QFrame()
        self.form.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.formLayout = QFormLayout(self.form)

        self.position_label = QLabel('Position', self.form)
        self.position_choice = QComboBox(self.form)
        self.formLayout.addRow(self.position_label, self.position_choice)

        self.list_view = QListView(self.form)
        self.list_model = QStringListModel()
        self.list_view.setModel(self.list_model)

        self.formLayout.addRow(self.list_view)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Close | QDialogButtonBox.Ok, self)
        self.button_box.setCenterButtons(True)

        self.formLayout.addRow(self.button_box)

        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.close)

        self.setWidget(self.form)

        parent.addDockWidget(Qt.LeftDockWidgetArea, self, Qt.Vertical)

    def accept(self):
        """
        Change the window title when a new position has been chosen
        """
        self.setWindowTitle(f'Results in DB example plugin/{self.position_choice.currentText()}')
