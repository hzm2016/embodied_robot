from PyQt5 import QtCore, QtWidgets

class MessageBubble(QtWidgets.QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("messageBubble")
        self.setStyleSheet(
            "QFrame#messageBubble {background-color: #e6f0ff; border-radius: 18px; border: 1px solid #c2d6ff;}"
            "QLabel {font-size: 30px; color: #0f172a; background-color: #e6f0ff;}"
        )
        self.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.setFixedWidth(640)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(18, 12, 18, 12)
        layout.setSpacing(4)

        self.label = QtWidgets.QLabel()
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.label.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        self.label.setSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Minimum)
        self.label.setFixedWidth(604) # 气泡宽度
        layout.addWidget(self.label)

    def append_text(self, text):
        self.label.setText(self.label.text() + text)

class DialogWindow(QtWidgets.QWidget):
    message_received = QtCore.pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("小凯医生")
        self.resize(700, 2000) # 窗口大小
        self.setStyleSheet("background-color: #f5f6fb;")

        self.current_bubble = None

        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(24, 24, 24, 24)
        main_layout.setSpacing(12)

        header = QtWidgets.QLabel("手术图像分析助手")
        header.setStyleSheet("font-size: 30px; font-weight: 600; color: #0f172a;")
        main_layout.addWidget(header)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea {border: none;}")
        main_layout.addWidget(self.scroll_area)

        self.scroll_content = QtWidgets.QWidget()
        self.scroll_layout = QtWidgets.QVBoxLayout(self.scroll_content)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setSpacing(18)

        self.placeholder = QtWidgets.QLabel("等待图片更新…")
        self.placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self.placeholder.setStyleSheet("color: #6b7280; font-size: 14px; padding: 24px 0;")
        self.scroll_layout.addWidget(self.placeholder)

        self.bottom_spacer = QtWidgets.QSpacerItem(
            0,
            0,
            QtWidgets.QSizePolicy.Minimum,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.scroll_layout.addItem(self.bottom_spacer)
        self.scroll_area.setWidget(self.scroll_content)

        self.message_received.connect(self._handle_stream_chunk)

    @QtCore.pyqtSlot(str)
    def _handle_stream_chunk(self, chunk):
        text = chunk or ""
        stripped = text.strip()

        if stripped.startswith("【检测到新图片"):
            self._add_status_chip(stripped)
            self.current_bubble = None
            return

        if stripped.startswith("【分析完成"):
            self._add_status_chip(stripped)
            self.current_bubble = None
            return

        if stripped.startswith("图片不存在"):
            self._add_status_chip(stripped)
            self.current_bubble = None
            return

        if not stripped and not self.current_bubble:
            return

        if not self.current_bubble:
            self._create_bubble()

        self.current_bubble.append_text(text)
        self._scroll_to_bottom_later()

    def _ensure_placeholder_hidden(self):
        if self.placeholder:
            self.placeholder.hide()
            self.placeholder = None

    def _create_bubble(self, initial_text=""):
        self._ensure_placeholder_hidden()
        bubble = MessageBubble()
        if initial_text:
            bubble.append_text(initial_text)

        row_container = QtWidgets.QWidget()
        row_layout = QtWidgets.QHBoxLayout(row_container)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        row_layout.addWidget(bubble, 0, QtCore.Qt.AlignLeft)
        row_layout.addStretch(1)

        insert_pos = max(0, self.scroll_layout.count() - 1)
        self.scroll_layout.insertWidget(insert_pos, row_container)
        self.current_bubble = bubble
        self._scroll_to_bottom_later()

    def _add_status_chip(self, message):
        self._ensure_placeholder_hidden()
        chip = QtWidgets.QLabel(message)
        chip.setAlignment(QtCore.Qt.AlignCenter)
        chip.setStyleSheet("color: #475569; font-size: 18px; padding: 4px 0;")
        insert_pos = max(0, self.scroll_layout.count() - 1)
        self.scroll_layout.insertWidget(insert_pos, chip)
        self._scroll_to_bottom_later()

    def _scroll_to_bottom_later(self):
        QtCore.QTimer.singleShot(0, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        self.scroll_area.verticalScrollBar().setValue(
            self.scroll_area.verticalScrollBar().maximum()
        )
