import win32api as wapi
import time

charList = []
for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ 123456789,.'£$/\\":
    charList.append(ord(char))


KEY_LEFT = 37
KEY_UP = 38
KEY_RIGHT = 39
KEY_DOWN = 40
keyList = [KEY_LEFT, KEY_RIGHT, KEY_UP, KEY_DOWN]


# 专门检测char的
def char_check():
    keys = []
    for key in charList:
        # if wapi.GetAsyncKeyState(ord(key)):
        if wapi.GetAsyncKeyState(key):  # 这个只针对于上下左右键
            keys.append(key)
    return keys


# 专门检测上下左右键的
def key_check():
    keys = []
    for key in keyList:
        # if wapi.GetAsyncKeyState(ord(key)):
        if wapi.GetAsyncKeyState(key):  # 这个只针对于上下左右键
            keys.append(key)
    return keys


# 返回一个hot array
def key_check_hot():
    out = [0, 0, 0, 0]
    for index, key in enumerate(keyList):
        if wapi.GetAsyncKeyState(key):  # 这个只针对于上下左右键
            out[index] = 1
    return out