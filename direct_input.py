import ctypes
import time

SendInput = ctypes.windll.user32.SendInput
# dwExtraInfo 在 64 位 Win32 API 中必须为 ULONG_PTR 整数类型（通常为 0）
# 严禁传入 Python 堆内存指针，否则 Steam 全局键盘钩子 (WH_KEYBOARD_LL) 将其当作回调跳转导致 0xc0000005 闪退！
ULONG_PTR = ctypes.c_size_t

class KeyBdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", ctypes.c_ushort),
        ("wScan", ctypes.c_ushort),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR)
    ]

class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", ctypes.c_ulong),
        ("wParamL", ctypes.c_short),
        ("wParamH", ctypes.c_ushort)
    ]

class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ULONG_PTR)
    ]

class Input_I(ctypes.Union):
    _fields_ = [
        ("ki", KeyBdInput),
        ("mi", MouseInput),
        ("hi", HardwareInput)
    ]

class Input(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_ulong),
        ("ii", Input_I)
    ]

SendInput.argtypes = [ctypes.c_uint, ctypes.POINTER(Input), ctypes.c_int]
SendInput.restype = ctypes.c_uint

INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
KEYEVENTF_EXTENDEDKEY = 0x0001

# 硬件扫描码 (Scan Code, Is_Extended)
# 剧情确认 / 对话翻页
SCAN_SPACE = (0x39, False) # 空格 [Space]

# 玩家 1（安全键位：彻底杜绝 Alt/Shift 冲突）
SCAN_W = (0x11, False) # 向上 [W]
SCAN_A = (0x1E, False) # 向左 [A]
SCAN_S = (0x1F, False) # 向下 [S]
SCAN_D = (0x20, False) # 向右 [D]
SCAN_J = (0x24, False) # 捡起/放下 [J]
SCAN_K = (0x25, False) # 切碎/投掷 [K]
SCAN_L = (0x26, False) # 加速冲刺 [L]

# 玩家 2（安全键位：方向键 + 小键盘 1/2/3）
SCAN_UP = (0x48, True)       # 向上 [上]
SCAN_LEFT = (0x4B, True)     # 向左 [左]
SCAN_DOWN = (0x50, True)     # 向下 [下]
SCAN_RIGHT = (0x4D, True)    # 向右 [右]
SCAN_NUM1 = (0x4F, False)    # 捡起/放下 [小键盘 1]
SCAN_NUM2 = (0x50, False)    # 切碎/投掷 [小键盘 2]
SCAN_NUM3 = (0x51, False)    # 加速冲刺 [小键盘 3]

def press_key(key_def):
    scan_code, is_extended = key_def
    flags = KEYEVENTF_SCANCODE
    if is_extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, scan_code, flags, 0, 0)
    x = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def release_key(key_def):
    scan_code, is_extended = key_def
    flags = KEYEVENTF_SCANCODE | KEYEVENTF_KEYUP
    if is_extended:
        flags |= KEYEVENTF_EXTENDEDKEY
    ii_ = Input_I()
    ii_.ki = KeyBdInput(0, scan_code, flags, 0, 0)
    x = Input(ctypes.c_ulong(INPUT_KEYBOARD), ii_)
    SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

class KeyController:
    def __init__(self):
        self.pressed_keys = set()

    def set_key(self, key_def, is_pressed):
        if is_pressed:
            if key_def not in self.pressed_keys:
                press_key(key_def)
                self.pressed_keys.add(key_def)
        else:
            if key_def in self.pressed_keys:
                release_key(key_def)
                self.pressed_keys.discard(key_def)

    def release_all(self):
        for k in list(self.pressed_keys):
            release_key(k)
        self.pressed_keys.clear()
