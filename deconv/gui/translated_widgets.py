"""Qt widget wrappers that retain English source text and retranslate in place."""
from __future__ import annotations

from typing import Any, Iterable, Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QAction as _QAction,
    QCheckBox as _QCheckBox,
    QComboBox as _QComboBox,
    QFileDialog as _QFileDialog,
    QFormLayout as _QFormLayout,
    QGroupBox as _QGroupBox,
    QInputDialog as _QInputDialog,
    QLabel as _QLabel,
    QLineEdit as _QLineEdit,
    QMessageBox as _QMessageBox,
    QPushButton as _QPushButton,
    QSlider as _QSlider,
    QStatusBar as _QStatusBar,
    QTabWidget as _QTabWidget,
    QTextEdit as _QTextEdit,
)

from deconv.i18n import get_language, register_retranslator, translate


class _TextMixin:
    _source_text: str
    _source_tooltip: str

    def _init_translation(self) -> None:
        self._source_text = ""
        self._source_tooltip = ""
        register_retranslator(self.retranslate)

    def setToolTip(self, text: str) -> None:  # type: ignore[override]
        self._source_tooltip = str(text or "")
        super().setToolTip(translate(self._source_tooltip))

    def retranslate(self) -> None:
        if self._source_tooltip:
            super().setToolTip(translate(self._source_tooltip))


class QLabel(_TextMixin, _QLabel):
    def __init__(self, text: str = "", parent=None, flags=Qt.WindowFlags()) -> None:
        _QLabel.__init__(self, "", parent, flags)
        self._init_translation()
        self.setText(text)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._source_text = str(text or "")
        _QLabel.setText(self, translate(self._source_text))

    def retranslate(self) -> None:
        _QLabel.setText(self, translate(self._source_text))
        _TextMixin.retranslate(self)


class QPushButton(_TextMixin, _QPushButton):
    def __init__(self, text: str = "", parent=None) -> None:
        _QPushButton.__init__(self, "", parent)
        self._init_translation()
        self.setText(text)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._source_text = str(text or "")
        _QPushButton.setText(self, translate(self._source_text))

    def retranslate(self) -> None:
        _QPushButton.setText(self, translate(self._source_text))
        _TextMixin.retranslate(self)


class QCheckBox(_TextMixin, _QCheckBox):
    def __init__(self, text: str = "", parent=None) -> None:
        _QCheckBox.__init__(self, "", parent)
        self._init_translation()
        self.setText(text)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._source_text = str(text or "")
        _QCheckBox.setText(self, translate(self._source_text))

    def retranslate(self) -> None:
        _QCheckBox.setText(self, translate(self._source_text))
        _TextMixin.retranslate(self)


class QGroupBox(_TextMixin, _QGroupBox):
    def __init__(self, title: str = "", parent=None) -> None:
        _QGroupBox.__init__(self, "", parent)
        self._init_translation()
        self.setTitle(title)

    def setTitle(self, title: str) -> None:  # type: ignore[override]
        self._source_text = str(title or "")
        _QGroupBox.setTitle(self, translate(self._source_text))

    def retranslate(self) -> None:
        _QGroupBox.setTitle(self, translate(self._source_text))
        _TextMixin.retranslate(self)


class QAction(_TextMixin, _QAction):
    def __init__(self, text: str = "", parent=None) -> None:
        _QAction.__init__(self, "", parent)
        self._init_translation()
        self.setText(text)

    def setText(self, text: str) -> None:  # type: ignore[override]
        self._source_text = str(text or "")
        _QAction.setText(self, translate(self._source_text))

    def retranslate(self) -> None:
        _QAction.setText(self, translate(self._source_text))
        _TextMixin.retranslate(self)


class QLineEdit(_TextMixin, _QLineEdit):
    def __init__(self, *args, **kwargs) -> None:
        _QLineEdit.__init__(self, *args, **kwargs)
        self._init_translation()
        self._source_placeholder = ""

    def setPlaceholderText(self, text: str) -> None:  # type: ignore[override]
        self._source_placeholder = str(text or "")
        _QLineEdit.setPlaceholderText(self, translate(self._source_placeholder))

    def retranslate(self) -> None:
        if self._source_placeholder:
            _QLineEdit.setPlaceholderText(self, translate(self._source_placeholder))
        _TextMixin.retranslate(self)


class QSlider(_TextMixin, _QSlider):
    def __init__(self, *args, **kwargs) -> None:
        _QSlider.__init__(self, *args, **kwargs)
        self._init_translation()


class QTextEdit(_TextMixin, _QTextEdit):
    def __init__(self, *args, **kwargs) -> None:
        _QTextEdit.__init__(self, *args, **kwargs)
        self._init_translation()
        self._source_lines = []

    def append(self, text: str) -> None:  # type: ignore[override]
        source = str(text or "")
        self._source_lines.append(source)
        _QTextEdit.append(self, translate(source))

    def clear(self) -> None:  # type: ignore[override]
        self._source_lines.clear()
        _QTextEdit.clear(self)

    def setPlainText(self, text: str) -> None:  # type: ignore[override]
        self._source_lines = str(text or "").splitlines()
        _QTextEdit.setPlainText(self, "\n".join(translate(line) for line in self._source_lines))

    def retranslate(self) -> None:
        if self._source_lines:
            _QTextEdit.setPlainText(self, "\n".join(translate(line) for line in self._source_lines))
        _TextMixin.retranslate(self)


class QComboBox(_TextMixin, _QComboBox):
    SOURCE_ROLE = Qt.UserRole + 73

    def __init__(self, *args, **kwargs) -> None:
        _QComboBox.__init__(self, *args, **kwargs)
        self._init_translation()

    def addItem(self, text: str, userData: Any = None) -> None:  # type: ignore[override]
        source = str(text)
        _QComboBox.addItem(self, translate(source), userData)
        self.setItemData(self.count() - 1, source, self.SOURCE_ROLE)

    def addItems(self, texts: Iterable[str]) -> None:  # type: ignore[override]
        for text in texts:
            self.addItem(str(text))

    def sourceText(self, index: int) -> str:
        source = self.itemData(index, self.SOURCE_ROLE)
        return str(source) if source is not None else _QComboBox.itemText(self, index)

    def currentText(self) -> str:  # type: ignore[override]
        return self.sourceText(self.currentIndex()) if self.currentIndex() >= 0 else ""

    def findText(self, text: str, flags=Qt.MatchExactly | Qt.MatchCaseSensitive) -> int:  # type: ignore[override]
        target = str(text)
        for index in range(self.count()):
            if self.sourceText(index) == target:
                return index
        return _QComboBox.findText(self, translate(target), flags)

    def setCurrentText(self, text: str) -> None:  # type: ignore[override]
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)
        else:
            _QComboBox.setCurrentText(self, translate(str(text)))

    def retranslate(self) -> None:
        index = self.currentIndex()
        blocked = self.blockSignals(True)
        for i in range(self.count()):
            _QComboBox.setItemText(self, i, translate(self.sourceText(i)))
        self.setCurrentIndex(index)
        self.blockSignals(blocked)
        _TextMixin.retranslate(self)


class QFormLayout(_QFormLayout):
    def addRow(self, *args) -> None:  # type: ignore[override]
        if args and isinstance(args[0], str):
            args = (QLabel(args[0]),) + args[1:]
        _QFormLayout.addRow(self, *args)

    def insertRow(self, row: int, *args) -> None:  # type: ignore[override]
        if args and isinstance(args[0], str):
            args = (QLabel(args[0]),) + args[1:]
        _QFormLayout.insertRow(self, row, *args)


class QTabWidget(_QTabWidget):
    SOURCE_ROLE = Qt.UserRole + 74

    def __init__(self, *args, **kwargs) -> None:
        _QTabWidget.__init__(self, *args, **kwargs)
        register_retranslator(self.retranslate)

    def addTab(self, widget, label: str) -> int:  # type: ignore[override]
        index = _QTabWidget.addTab(self, widget, translate(label))
        self.tabBar().setTabData(index, str(label))
        return index

    def retranslate(self) -> None:
        for index in range(self.count()):
            source = self.tabBar().tabData(index)
            if source is not None:
                self.setTabText(index, translate(str(source)))


class QStatusBar(_QStatusBar):
    def __init__(self, *args, **kwargs) -> None:
        _QStatusBar.__init__(self, *args, **kwargs)
        self._source_message = ""
        register_retranslator(self.retranslate)

    def showMessage(self, message: str, timeout: int = 0) -> None:  # type: ignore[override]
        self._source_message = str(message or "")
        _QStatusBar.showMessage(self, translate(self._source_message), timeout)

    def retranslate(self) -> None:
        if self._source_message:
            _QStatusBar.showMessage(self, translate(self._source_message))


class QMessageBox:
    Yes = _QMessageBox.Yes
    No = _QMessageBox.No
    Ok = _QMessageBox.Ok
    Cancel = _QMessageBox.Cancel

    @staticmethod
    def _standard(parent, icon, title, text, buttons=_QMessageBox.Ok, default_button=_QMessageBox.NoButton):
        box = _QMessageBox(icon, translate(title), translate(text), buttons, parent)
        if default_button != _QMessageBox.NoButton:
            box.setDefaultButton(default_button)
        labels = {
            _QMessageBox.Yes: translate("Yes"),
            _QMessageBox.No: translate("No"),
            _QMessageBox.Ok: translate("OK"),
            _QMessageBox.Cancel: translate("Cancel"),
        }
        for standard, label in labels.items():
            button = box.button(standard)
            if button is not None:
                button.setText(label)
        return box.exec_()

    @staticmethod
    def question(parent, title, text, buttons=_QMessageBox.Yes | _QMessageBox.No, defaultButton=_QMessageBox.NoButton):
        return QMessageBox._standard(parent, _QMessageBox.Question, title, text, buttons, defaultButton)

    @staticmethod
    def warning(parent, title, text, buttons=_QMessageBox.Ok, defaultButton=_QMessageBox.NoButton):
        return QMessageBox._standard(parent, _QMessageBox.Warning, title, text, buttons, defaultButton)

    @staticmethod
    def information(parent, title, text, buttons=_QMessageBox.Ok, defaultButton=_QMessageBox.NoButton):
        return QMessageBox._standard(parent, _QMessageBox.Information, title, text, buttons, defaultButton)

    @staticmethod
    def critical(parent, title, text, buttons=_QMessageBox.Ok, defaultButton=_QMessageBox.NoButton):
        return QMessageBox._standard(parent, _QMessageBox.Critical, title, text, buttons, defaultButton)


class QFileDialog:
    @staticmethod
    def getOpenFileName(parent=None, caption="", directory="", filter="", initialFilter="", options=_QFileDialog.Options()):
        return _QFileDialog.getOpenFileName(parent, translate(caption), directory, translate(filter), initialFilter, options)

    @staticmethod
    def getSaveFileName(parent=None, caption="", directory="", filter="", initialFilter="", options=_QFileDialog.Options()):
        return _QFileDialog.getSaveFileName(parent, translate(caption), directory, translate(filter), initialFilter, options)


class QInputDialog:
    @staticmethod
    def getItem(parent, title, label, items, current=0, editable=True, flags=Qt.WindowFlags(), inputMethodHints=Qt.ImhNone):
        translated_items = [translate(str(item)) for item in items]
        return _QInputDialog.getItem(parent, translate(title), translate(label), translated_items, current, editable, flags, inputMethodHints)
