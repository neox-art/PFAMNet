"""Styles scoped to the dock, leaving the QGIS application theme intact."""

STYLE = '''
QWidget#parcelWorkbench { background: #f8faf9; color: #24332d; }
QWidget#parcelWorkbench QLabel { color: #45554e; }
QWidget#parcelWorkbench QLabel#brand { color: #173d31; font-size: 18px; font-weight: 600; }
QWidget#parcelWorkbench QLabel#model { color: #68756e; font-size: 11px; }
QWidget#parcelWorkbench QLabel[section="true"] { color: #223b30; font-weight: 600;
    border-bottom: 1px solid #e2e8e4; padding: 9px 0 7px 0; }
QWidget#parcelWorkbench QTabWidget::pane { border: 0; background: white; }
QWidget#parcelWorkbench QTabBar::tab { background: #f8faf9; color: #69766f;
    padding: 10px 16px; border-bottom: 2px solid transparent; }
QWidget#parcelWorkbench QTabBar::tab:selected { color: #16684c; border-bottom: 2px solid #198561; }
QWidget#parcelWorkbench QScrollArea, QWidget#parcelWorkbench QWidget#page { background: white; }
QWidget#parcelWorkbench QLineEdit, QWidget#parcelWorkbench QComboBox,
QWidget#parcelWorkbench QAbstractSpinBox { background: white; color: #25372e;
    border: 1px solid #d4ddd7; border-radius: 3px; min-height: 28px; padding: 0 6px; }
QWidget#parcelWorkbench QLineEdit:focus, QWidget#parcelWorkbench QComboBox:focus { border: 1px solid #268568; }
QWidget#parcelWorkbench QPushButton, QWidget#parcelWorkbench QToolButton { color: #344a40;
    background: #f6f8f7; border: 1px solid #d7e0da; border-radius: 3px; padding: 5px; }
QWidget#parcelWorkbench QPushButton:hover, QWidget#parcelWorkbench QToolButton:hover {
    background: #eaf3ee; border-color: #8fb6a2; }
QWidget#parcelWorkbench QToolButton:checked { background: #dcefe5; border: 1px solid #258a60; }
QWidget#parcelWorkbench QPushButton#primary { background: #197953; color: white; border-color: #197953; }
QWidget#parcelWorkbench QPushButton#primary:hover { background: #126343; }
QWidget#parcelWorkbench QPushButton:disabled { background: #f1f4f2; color: #a2aca6; border-color: #e3e8e5; }
QWidget#parcelWorkbench QPushButton#primary:disabled { background: #e7ede9; color: #8b9c91; border-color: #dce5df; }
QWidget#parcelWorkbench QListWidget { background: white; color: #354b3e; border: 1px solid #dce4df;
    border-radius: 3px; outline: 0; }
QWidget#parcelWorkbench QListWidget::item { padding: 9px 8px; border-bottom: 1px solid #edf1ee; }
QWidget#parcelWorkbench QListWidget::item:selected { background: #deeee7; color: #14593c; }
QWidget#parcelWorkbench QSlider::groove:horizontal { height: 4px; background: #e2e8e4; border-radius: 2px; }
QWidget#parcelWorkbench QSlider::sub-page:horizontal { background: #528dba; }
QWidget#parcelWorkbench QSlider::handle:horizontal { background: #fff; border: 1px solid #528dba;
    width: 12px; margin: -5px 0; border-radius: 6px; }
QWidget#parcelWorkbench QLabel#status { padding: 8px; background: #edf3ef; border-radius: 3px; }
'''
