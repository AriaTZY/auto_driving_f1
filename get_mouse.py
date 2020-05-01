import win32api,win32con,time
from ctypes import *

class POINT(Structure):
    _fields_ = [("x", c_ulong), ("y", c_ulong)]

def mouse_move(x, y):
    """
    移动鼠标到（x，y）坐标
    :param x:
    :param y:
    """
    windll.user32.SetCursorPos(x, y)

def get_mouse_point():
    """
    获取鼠标当前位置坐标
    """
    po = POINT()
    windll.user32.GetCursorPos(byref(po))
    return int(po.x), int(po.y)

def mouse_click(x=None, y=None):
    """
    根据屏幕坐标进行鼠标单击操作
    :param x:
    :param y:
    """
    if not x is None and not y is None:
        mouse_move(x, y)
        time.sleep(0.05)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

def mouse_dclick(x=None, y=None):
    """
    根据屏幕坐标进行鼠标双击操作
    :param x:
    :param y:
    """
    if not x is None and not y is None:
        mouse_move(x, y)
        time.sleep(0.05)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)



while True:
    x, y = get_mouse_point()
    print(x, y)
    time.sleep(0.9)