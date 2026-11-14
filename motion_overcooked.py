import os
import sys
import time
import math
import ctypes
import threading
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 屏蔽底层 C++ 警告
os.environ['GLOG_minloglevel'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    kernel32 = ctypes.windll.kernel32
    hStdin = kernel32.GetStdHandle(-10)
    mode = ctypes.c_ulong()
    kernel32.GetConsoleMode(hStdin, ctypes.byref(mode))
    mode.value &= ~0x0040
    kernel32.SetConsoleMode(hStdin, mode)
except Exception:
    pass

def silence_stderr():
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(devnull, 2)
        os.close(devnull)
        return old_stderr
    except Exception:
        return None

def restore_stderr(old_stderr):
    if old_stderr is not None:
        try:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except Exception:
            pass

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

_old_err = silence_stderr()
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
restore_stderr(_old_err)

from direct_input import (
    KeyController,
    press_key, release_key,
    SCAN_SPACE,
    SCAN_W, SCAN_A, SCAN_S, SCAN_D, SCAN_J, SCAN_K, SCAN_L,
    SCAN_UP, SCAN_LEFT, SCAN_DOWN, SCAN_RIGHT, SCAN_NUM1, SCAN_NUM2, SCAN_NUM3
)

FONT_PATH = r"C:\Windows\Fonts\msyh.ttc"
try:
    font_bold = ImageFont.truetype(FONT_PATH, 15)
    font_sub = ImageFont.truetype(FONT_PATH, 11)
    font_btn = ImageFont.truetype(FONT_PATH, 12)
    font_banner = ImageFont.truetype(FONT_PATH, 20)
    font_hint = ImageFont.truetype(FONT_PATH, 12)
except Exception:
    font_bold = ImageFont.load_default()
    font_sub = font_bold
    font_btn = font_bold
    font_banner = font_bold
    font_hint = font_bold

CAMERA_INDEX = 0
MODEL_PATH = os.path.join(SCRIPT_DIR, "pose_landmarker_lite.task")

# 经 EMA 滤波后的稳态死区阈值（下蹲为下，踮脚为上；前倾为上，后仰为下）
TILT_X_THRESH = 0.18        # 左右侧倾死区（身体左倾=向左走，右倾=向右走）
REL_Y_UP_ENTER = -0.075     # 拔高/踮脚进入阈值（向上走）
REL_Y_UP_EXIT = -0.045      # 拔高退出阈值
REL_Y_DN_ENTER = 0.050      # 自然微屈膝即可向下走（轻松触发，告别费力深蹲！）
REL_Y_DN_EXIT = 0.025       # 屈膝退出阈值
LEAN_PITCH_UP = 0.050       # 身体微前倾 -> 亦可触发向上走 [W]！
LEAN_PITCH_DN = -0.050      # 身体微后仰 -> 亦可触发向下走 [S]！
DASH_PITCH_THRESH = 0.115   # 爆发深前倾冲刺阈值

P1_KEYS = {
    'UP': SCAN_W,            # [W]
    'DOWN': SCAN_S,          # [S]
    'LEFT': SCAN_A,          # [A]
    'RIGHT': SCAN_D,         # [D]
    'PICK': SCAN_J,          # 捡起/放下 [J]
    'CHOP': SCAN_K,          # 切碎/投掷 [K]
    'DASH': SCAN_L,          # 加速冲刺 [L]
}

P2_KEYS = {
    'UP': SCAN_UP,           # [上]
    'DOWN': SCAN_DOWN,       # [下]
    'LEFT': SCAN_LEFT,       # [左]
    'RIGHT': SCAN_RIGHT,     # [右]
    'PICK': SCAN_NUM1,       # 捡起/放下 [小键盘 1]
    'CHOP': SCAN_NUM2,       # 切碎/投掷 [小键盘 2]
    'DASH': SCAN_NUM3,       # 加速冲刺 [小键盘 3]
}

scheduled_releases = {}

def async_tap_key(key_def, duration=0.10):
    if key_def not in scheduled_releases:
        press_key(key_def)
    scheduled_releases[key_def] = max(scheduled_releases.get(key_def, 0.0), time.time() + duration)

def update_scheduled_keys(now):
    for k in list(scheduled_releases.keys()):
        if now >= scheduled_releases[k]:
            release_key(k)
            scheduled_releases.pop(k, None)

user32 = ctypes.windll.user32
def is_global_key_pressed(vk_code):
    return (user32.GetAsyncKeyState(vk_code) & 0x8000) != 0

VK_CONTROL = 0x11
VK_Q = 0x51
VK_C = 0x43    # 随时按 [C] 秒级重置对齐站姿
VK_F9 = 0x78   # 一键切换【单屏画中画置顶小窗】与【双屏高清大窗】
VK_F10 = 0x79  # 退出

class FastCamera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _reader_loop(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.flip(frame, 1) # 镜面模式
                with self.lock:
                    self.frame = frame
                    self.ret = ret
            else:
                time.sleep(0.005)

    def read_latest(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def release(self):
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()


last_space_trigger_time = 0.0
space_anim_until = 0.0
recalib_anim_until = 0.0
recalib_msg = ""

class PlayerTracker:
    def __init__(self, player_id, keys, color):
        self.player_id = player_id
        self.keys = keys
        self.color = color
        
        # 3D 全身自适应基准
        self.base_body_y = None
        self.base_shoulder_w = None
        self.base_z_forward = None
        self.base_tilt_x = None
        
        # 姿态充能校准状态机（绝不再赶时间！走到位置双手叉腰1秒即可锁定）
        self.calib_hold_start = None
        self.calib_progress = 0.0
        self.is_calibrated = False

        # EMA 低通平滑值（彻底滤除高频视觉抖动噪点）
        self.smooth_tilt_x = 0.0
        self.smooth_rel_y = 0.0
        self.smooth_pitch = 0.0
        
        # 状态机锁存（用于迟滞死区）
        self.is_moving_up = False
        self.is_moving_down = False
        self.squat_exit_time = 0.0
        self.squat_exit_time = 0.0  # 下蹲起立时间戳（防回弹误判）
        self.is_hand_lifted = False
        self.is_l_raised = False
        self.is_r_raised = False
        self.hand_lowered_time = 0.0
        self.is_hands_together = False
        
        # 切菜行程位移队列
        self.lw_history = []
        self.rw_history = []
        
        self.last_chop_time = 0
        self.last_pick_time = 0
        self.last_dash_time = 0
        self.chop_sustain_until = 0.0
        self.last_chop_tap_time = 0.0
        self.active_actions = set()

        # 4向走位单点与离散脉冲发生器（初次命中单点，持续按住节奏点按，彻底告别长按失控滑步！）
        self.dir_tap_state = {
            'LEFT': {'active': False, 'first_hit': 0.0, 'last_tap': 0.0},
            'RIGHT': {'active': False, 'first_hit': 0.0, 'last_tap': 0.0},
            'UP': {'active': False, 'first_hit': 0.0, 'last_tap': 0.0},
            'DOWN': {'active': False, 'first_hit': 0.0, 'last_tap': 0.0},
        }
        self.action_visual_until = {}

    def trigger_tap(self, act_name, key_def, now, duration=0.08):
        async_tap_key(key_def, duration=duration)
        self.active_actions.add(act_name)
        self.action_visual_until[act_name] = now + 0.16

    def process_dir_tap(self, act_name, is_triggered, key_def, now):
        st = self.dir_tap_state[act_name]
        if is_triggered:
            if not st['active']:
                # 初次命中：立即触发清脆的【单点】！
                st['active'] = True
                st['first_hit'] = now
                st['last_tap'] = now
                self.trigger_tap(act_name, key_def, now, duration=0.08)
            else:
                # 持续保持该姿态：首发延迟 0.26s 后，以 0.13s 节奏输出离散单点脉冲
                # 绝不长按锁死按键，角色不会失控狂奔滑移！
                if (now - st['first_hit'] > 0.26) and (now - st['last_tap'] > 0.13):
                    self.trigger_tap(act_name, key_def, now, duration=0.08)
                    st['last_tap'] = now
        else:
            st['active'] = False

    def check_calib_pose(self, landmarks):
        if not landmarks or len(landmarks) < 17:
            return False
        ls, rs = landmarks[11], landmarks[12]
        le, re = landmarks[13], landmarks[14]
        lw, rw = landmarks[15], landmarks[16]

        sm_x = (ls.x + rs.x) / 2.0
        sm_y = (ls.y + rs.y) / 2.0
        shoulder_w = max(0.06, math.hypot(rs.x - ls.x, rs.y - ls.y))

        # 专属姿态 1：双手叉腰（双手搭在腰间，双肘向两侧自然张开，游戏中绝对不会出现的动作）
        wrists_at_waist = (lw.y > sm_y + 0.08) and (lw.y < sm_y + 0.45) and (rw.y > sm_y + 0.08) and (rw.y < sm_y + 0.45)
        wrists_near_body = (abs(lw.x - sm_x) < 1.15 * shoulder_w) and (abs(rw.x - sm_x) < 1.15 * shoulder_w)
        elbows_flared = (abs(le.x - sm_x) > 0.58 * shoulder_w) and (abs(re.x - sm_x) > 0.58 * shoulder_w)
        is_akimbo = wrists_at_waist and wrists_near_body and elbows_flared

        # 专属姿态 2：双臂侧平举 (T-Pose)
        wrists_at_shoulder = (abs(lw.y - sm_y) < 0.16) and (abs(rw.y - sm_y) < 0.16)
        arms_extended = (abs(lw.x - sm_x) > 1.05 * shoulder_w) and (abs(rw.x - sm_x) > 1.05 * shoulder_w)
        is_t_pose = wrists_at_shoulder and arms_extended

        return is_akimbo or is_t_pose

    def lock_baseline(self, landmarks):
        if not landmarks:
            return
        nose = landmarks[0]
        ls, rs = landmarks[11], landmarks[12]
        lh, rh = landmarks[23], landmarks[24]
        sm_x = (ls.x + rs.x) / 2.0
        sm_y = (ls.y + rs.y) / 2.0
        sm_z = (ls.z + rs.z) / 2.0
        shoulder_w = max(0.06, math.hypot(rs.x - ls.x, rs.y - ls.y))

        # 统一使用双肩中点作为绝对躯干基准（永远在镜头内，绝不受下半身视野切割影响）
        body_y = sm_y

        self.base_tilt_x = (nose.x - sm_x) / shoulder_w
        self.base_z_forward = sm_z - nose.z
        self.base_body_y = body_y
        self.base_shoulder_w = shoulder_w

        self.smooth_tilt_x = 0.0
        self.smooth_rel_y = 0.0
        self.smooth_pitch = 0.0
        self.is_moving_up = False
        self.is_moving_down = False
        self.squat_exit_time = 0.0
        for st in self.dir_tap_state.values():
            st['active'] = False
        self.chop_sustain_until = 0.0
        self.action_visual_until.clear()
        self.is_calibrated = True

    def update(self, landmarks, controller, now):
        global last_space_trigger_time, space_anim_until, recalib_anim_until, recalib_msg
        self.active_actions.clear()
        for act, until_t in list(self.action_visual_until.items()):
            if now < until_t:
                self.active_actions.add(act)
            else:
                del self.action_visual_until[act]

        if not landmarks:
            self.calib_hold_start = None
            self.calib_progress = 0.0
            for st in self.dir_tap_state.values():
                st['active'] = False
            return

        nose = landmarks[0]
        ls, rs = landmarks[11], landmarks[12]
        lw, rw = landmarks[15], landmarks[16]
        lh, rh = landmarks[23], landmarks[24]

        # -------------------------------------------------------------
        # 核心：姿态触发式站姿校准（双手叉腰/平举 0.85 秒即锁死中立位！）
        # -------------------------------------------------------------
        is_calib_pose = self.check_calib_pose(landmarks)
        if is_calib_pose:
            if self.calib_hold_start is None:
                self.calib_hold_start = now
            self.calib_progress = min(1.0, (now - self.calib_hold_start) / 0.85)
            if self.calib_progress >= 1.0:
                self.lock_baseline(landmarks)
                recalib_msg = f"🎯 P{self.player_id} 中立站姿已锁定！"
                recalib_anim_until = now + 1.5
                self.calib_hold_start = None
                self.calib_progress = 0.0
        else:
            self.calib_hold_start = None
            self.calib_progress = 0.0

        # 如果尚未校准，不发送任何按键（让玩家悠闲走位就座，绝不误触游戏！）
        if not self.is_calibrated:
            for st in self.dir_tap_state.values():
                st['active'] = False
            return

        sm_x = (ls.x + rs.x) / 2.0
        sm_y = (ls.y + rs.y) / 2.0
        sm_z = (ls.z + rs.z) / 2.0
        shoulder_w = max(0.06, math.hypot(rs.x - ls.x, rs.y - ls.y))

        body_y = sm_y

        # -------------------------------------------------------------
        # 0. 剧情翻页（必须双手真正击掌拍手 👏，单点触发）
        # -------------------------------------------------------------
        hands_dist = math.hypot(lw.x - rw.x, lw.y - rw.y)
        is_touching = (hands_dist < 0.075) and (lw.y < sm_y + 0.20 and rw.y < sm_y + 0.20)

        if is_touching and not self.is_hands_together:
            if now - last_space_trigger_time > 0.8:
                self.trigger_tap('SPACE', SCAN_SPACE, now, duration=0.08)
                last_space_trigger_time = now
                space_anim_until = now + 1.0

        if hands_dist > 0.15:
            self.is_hands_together = False
        elif is_touching:
            self.is_hands_together = True

        # -------------------------------------------------------------
        # 1. 走位控制：左右侧倾=左/右！下蹲=下！踮脚/拔高=上！（纯净单点脉冲）
        # -------------------------------------------------------------
        raw_tilt_x = (nose.x - sm_x) / shoulder_w
        raw_rel_y = (body_y - self.base_body_y) / shoulder_w
        raw_z_forward = sm_z - nose.z

        # 严禁动态学习 base_body_y，牢牢锁定叉腰校准时的黄金零位，彻底杜绝下蹲后基准线漂移！
        norm_tilt_x = raw_tilt_x - self.base_tilt_x
        raw_pitch = raw_z_forward - self.base_z_forward

        self.smooth_tilt_x = self.smooth_tilt_x * 0.65 + norm_tilt_x * 0.35
        self.smooth_rel_y = self.smooth_rel_y * 0.65 + raw_rel_y * 0.35
        self.smooth_pitch = self.smooth_pitch * 0.65 + raw_pitch * 0.35

        # 左右走位（使用纯净 X 轴）
        move_left = self.smooth_tilt_x < -TILT_X_THRESH
        move_right = self.smooth_tilt_x > TILT_X_THRESH

        # 侧倾几何沉降动态补偿：左右倾斜走位时双肩中点会自然倾斜下沉，动态加宽深蹲阈值，100% 杜绝左右走位触发向下！
        tilt_drop_margin = 0.25 * max(0.0, abs(self.smooth_tilt_x) - 0.10)
        curr_dn_enter = REL_Y_DN_ENTER + tilt_drop_margin
        curr_dn_exit = REL_Y_DN_EXIT + tilt_drop_margin

        # 1. 判定向下走（自然微屈膝下蹲 OR 身体微后仰，双触发机制，极其轻松顺畅！）
        is_squat_dn = (self.smooth_rel_y > curr_dn_enter)
        is_lean_dn = (self.smooth_pitch < LEAN_PITCH_DN)
        trigger_down_raw = is_squat_dn or is_lean_dn

        was_down = self.is_moving_down
        if not self.is_moving_down:
            if trigger_down_raw:
                self.is_moving_down = True
        else:
            if (self.smooth_rel_y < curr_dn_exit) and (self.smooth_pitch > LEAN_PITCH_DN + 0.02):
                self.is_moving_down = False

        # 下蹲起立防回弹保护：当下蹲退出（用户起立站直）的瞬间，立即记录时间戳并开启 0.55 秒防回弹锁！
        if was_down and not self.is_moving_down:
            self.squat_exit_time = now

        # 2. 判定向上走（拔高/踮脚 OR 身体微前倾）
        # 核心防回弹保护：下蹲起立后的 0.55 秒内绝对禁止触发向上走，彻底告别“蹲下起来误判向上”！
        rebound_guard_active = (now - self.squat_exit_time < 0.55)

        is_stand_up = (self.smooth_rel_y < REL_Y_UP_ENTER)
        is_lean_up = (self.smooth_pitch > LEAN_PITCH_UP)
        trigger_up_raw = (is_stand_up or is_lean_up) and not rebound_guard_active

        if not self.is_moving_up:
            if trigger_up_raw:
                self.is_moving_up = True
        else:
            if (self.smooth_rel_y > REL_Y_UP_EXIT and self.smooth_pitch < LEAN_PITCH_UP - 0.02) or rebound_guard_active:
                self.is_moving_up = False

        move_up = self.is_moving_up
        move_down = self.is_moving_down

        if move_up and move_down:
            move_up = False
            move_down = False

        # 核心变革：4向走位全部采用离散单点脉冲引擎！
        # 初次命中即刻单点触发，按住时平滑节奏输出，绝不长按锁死，角色不滑步、不失控！
        self.process_dir_tap('LEFT', move_left, self.keys['LEFT'], now)
        self.process_dir_tap('RIGHT', move_right, self.keys['RIGHT'], now)
        self.process_dir_tap('UP', move_up, self.keys['UP'], now)
        self.process_dir_tap('DOWN', move_down, self.keys['DOWN'], now)

        # -------------------------------------------------------------
        # 2. 捡起/放下（迟滞比较器：入态 y < sm_y - 0.07，退态 y > sm_y + 0.05）
        # -------------------------------------------------------------
        # 左手状态机
        if not self.is_l_raised:
            if (lw.y < sm_y - 0.07) and (hands_dist > 0.16):
                self.is_l_raised = True
        else:
            if (lw.y > sm_y + 0.05) or (hands_dist < 0.12):
                self.is_l_raised = False
                self.hand_lowered_time = now
                self.lw_history.clear()

        # 右手状态机
        if not self.is_r_raised:
            if (rw.y < sm_y - 0.07) and (hands_dist > 0.16):
                self.is_r_raised = True
        else:
            if (rw.y > sm_y + 0.05) or (hands_dist < 0.12):
                self.is_r_raised = False
                self.hand_lowered_time = now
                self.rw_history.clear()

        any_hand_raised = self.is_l_raised or self.is_r_raised
        single_hand_raised = self.is_l_raised ^ self.is_r_raised

        if single_hand_raised and not self.is_hand_lifted:
            if now - self.last_pick_time > 0.35:
                self.trigger_tap('PICK', self.keys['PICK'], now, duration=0.08)
                self.last_pick_time = now
            self.lw_history.clear()
            self.rw_history.clear()

        self.is_hand_lifted = single_hand_raised

        # -------------------------------------------------------------
        # 3. 切碎/投掷（手腕相对双肩相对位移判定，深蹲/踮脚/移动时100%锁死，杜绝误扔食材！）
        # -------------------------------------------------------------
        # 保护门禁：
        # 1. 任何一只手处于拿放高抬状态时 -> 锁定不可切菜！
        # 2. 正在移动走位（左/右）、深蹲（下）、踮脚（上）时 -> 锁定不可切菜/投掷！
        # 3. 放下手之后的 0.40 秒防落臂惯性保护期内 -> 锁定不可切菜！
        # 4. 距离上次拿放触发不足 0.40 秒内 -> 锁定不可切菜！
        is_moving_any = move_left or move_right or move_up or move_down or self.is_moving_up or self.is_moving_down
        can_chop = (not any_hand_raised) and (not is_moving_any) and \
                   (now - self.hand_lowered_time > 0.40) and \
                   (now - self.last_pick_time > 0.40)

        # 移动走位、深蹲、踮脚或举手时，立即瞬间打断切菜连击
        if is_moving_any or any_hand_raised:
            self.chop_sustain_until = 0.0
            self.action_visual_until.pop('CHOP', None)

        # 核心数学解耦：计算手腕【相对于双肩中心】的相对高度！
        # 下蹲或踮脚时全身同步升降，手腕相对双肩位移严格为 0，物理级彻底杜绝下蹲误扔东西！
        rel_lw = lw.y - sm_y
        rel_rw = rw.y - sm_y

        REL_CHOP_TOP = -0.02
        REL_CHOP_BOTTOM = 0.42

        # 只要手在胸前躯干活动空间内就记录相对轨迹；若手在向上蓄力起刀，自动重置队列以最高点作为起刀点
        if can_chop and (REL_CHOP_TOP < rel_lw < REL_CHOP_BOTTOM):
            if len(self.lw_history) > 0 and (rel_lw < self.lw_history[-1][0] - 0.010):
                self.lw_history.clear()
            self.lw_history.append((rel_lw, now))
        else:
            self.lw_history.clear()

        if can_chop and (REL_CHOP_TOP < rel_rw < REL_CHOP_BOTTOM):
            if len(self.rw_history) > 0 and (rel_rw < self.rw_history[-1][0] - 0.010):
                self.rw_history.clear()
            self.rw_history.append((rel_rw, now))
        else:
            self.rw_history.clear()

        self.lw_history = [(y_h, t_h) for (y_h, t_h) in self.lw_history if now - t_h <= 0.22]
        self.rw_history = [(y_h, t_h) for (y_h, t_h) in self.rw_history if now - t_h <= 0.22]

        is_chopping_stroke = False
        if can_chop:
            for hist in (self.lw_history, self.rw_history):
                if len(hist) >= 3:
                    downward_travel = 0.0
                    for idx in range(1, len(hist)):
                        dy = hist[idx][0] - hist[idx - 1][0]
                        if dy > 0:
                            downward_travel += dy

                    dt_h = hist[-1][1] - hist[0][1]
                    speed = downward_travel / max(0.01, dt_h)
                    net_disp = hist[-1][0] - hist[0][0]

                    # 敏捷相对下斩判定：手腕相对双肩向下挥砍 > 0.030，速度 > 0.25，且终点净下移 > 0.020
                    if downward_travel > 0.030 and speed > 0.25 and net_disp > 0.020:
                        is_chopping_stroke = True
                        hist.clear() # 触发一次即清空该手轨迹，杜绝重复统计
                        break

        # 核心创新：每次自然向下挥刀一击，自动激活 0.60 秒的高频持续切菜连点！
        if is_chopping_stroke:
            self.chop_sustain_until = now + 0.60

        # 切菜持续连点执行器：在 0.60 秒连击窗口期内，每 0.11 秒自动高频连点（约每秒 9 刀）！
        # 挥一下剁数刀，连着挥两刀即可丝滑切完一整个食材，彻底告别手臂挥断！
        if can_chop and (now < self.chop_sustain_until):
            if now - self.last_chop_tap_time >= 0.11:
                self.trigger_tap('CHOP', self.keys['CHOP'], now, duration=0.06)
                self.last_chop_tap_time = now

        # -------------------------------------------------------------
        # 4. 加速冲刺（仅在身体深前倾突进 > 0.082 时单点触发）
        # -------------------------------------------------------------
        is_dash = (self.smooth_pitch > DASH_PITCH_THRESH)
        if is_dash:
            if now - self.last_dash_time > 0.45:
                self.trigger_tap('DASH', self.keys['DASH'], now, duration=0.08)
                self.last_dash_time = now


FULL_SKELETON_PAIRS = [
    (11, 12),
    (0, 11),  (0, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27),
    (24, 26), (26, 28),
    (27, 29), (29, 31),
    (28, 30), (30, 32)
]

ALL_JOINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]

def draw_cyber_skeleton(img, pose, neon_color, w, h):
    if not pose:
        return
    overlay = img.copy()
    for (i1, i2) in FULL_SKELETON_PAIRS:
        if i1 < len(pose) and i2 < len(pose):
            p1, p2 = pose[i1], pose[i2]
            if p1.visibility > 0.35 and p2.visibility > 0.35:
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(overlay, (x1, y1), (x2, y2), neon_color, 6, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    for (i1, i2) in FULL_SKELETON_PAIRS:
        if i1 < len(pose) and i2 < len(pose):
            p1, p2 = pose[i1], pose[i2]
            if p1.visibility > 0.35 and p2.visibility > 0.35:
                x1, y1 = int(p1.x * w), int(p1.y * h)
                x2, y2 = int(p2.x * w), int(p2.y * h)
                cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 2, cv2.LINE_AA)

    for idx in ALL_JOINTS:
        if idx < len(pose):
            pt = pose[idx]
            if pt.visibility > 0.35:
                px, py = int(pt.x * w), int(pt.y * h)
                if idx in (15, 16, 31, 32):
                    cv2.circle(img, (px, py), 8, neon_color, 2, cv2.LINE_AA)
                    cv2.circle(img, (px, py), 4, (255, 255, 255), -1, cv2.LINE_AA)
                else:
                    cv2.circle(img, (px, py), 5, neon_color, -1, cv2.LINE_AA)
                    cv2.circle(img, (px, py), 2, (255, 255, 255), -1, cv2.LINE_AA)


def draw_cyber_hud(display_img, p1, p2, fps, now):
    global space_anim_until, recalib_anim_until, recalib_msg
    h, w = display_img.shape[:2]

    for y in range(135, h - 70, 20):
        cv2.line(display_img, (w // 2, y), (w // 2, y + 10), (80, 100, 120), 1, cv2.LINE_AA)

    pil_img = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img, 'RGBA')

    def draw_player_card(x0, y0, x1, y1, title, theme_color, bg_color, player_obj, is_p1):
        draw.rounded_rectangle([(x0, y0), (x1, y1)], radius=12, fill=bg_color, outline=theme_color, width=2)

        is_detected = player_obj.is_calibrated
        actions = player_obj.active_actions

        # 叉腰校准状态指示
        if not is_detected:
            if player_obj.calib_progress > 0.0:
                status_text = f"充能 {int(player_obj.calib_progress * 100)}%"
                status_col = (0, 255, 220)
            else:
                status_text = "○ 叉腰1秒校准"
                status_col = (255, 200, 50)
        else:
            status_text = "● 已对齐锁定"
            status_col = (0, 255, 150)

        draw.text((x0 + 12, y0 + 10), title, font=font_bold, fill=theme_color)
        draw.text((x1 - 85, y0 + 12), status_text, font=font_sub, fill=status_col)

        # 充能进度发光条
        if player_obj.calib_progress > 0.0:
            bar_w = int((x1 - x0 - 24) * player_obj.calib_progress)
            draw.rounded_rectangle([(x0 + 12, y0 + 30), (x0 + 12 + bar_w, y0 + 34)], radius=2,
                                   fill=(0, 255, 200, 255))

        dirs = [('LEFT', '←左'), ('UP', '↑上'), ('DOWN', '↓下'), ('RIGHT', '→右')]
        bx = x0 + 12
        by = y0 + 36
        for act, lbl in dirs:
            active = act in actions
            if active:
                fill_col = (0, 230, 80, 255)
                outline_col = (255, 255, 255, 255)
                txt_col = (0, 0, 0)
            else:
                fill_col = (40, 45, 55, 180)
                outline_col = (70, 75, 90, 200)
                txt_col = (180, 180, 190)
            draw.rounded_rectangle([(bx, by), (bx + 40, by + 26)], radius=6, fill=fill_col, outline=outline_col, width=1)
            draw.text((bx + 7, by + 4), lbl, font=font_btn, fill=txt_col)
            bx += 45

        bx = x0 + 12
        by = y0 + 68
        chop_lbl = "切碎 [K]" if is_p1 else "切碎 [小键2]"
        pick_lbl = "拿放 [J]" if is_p1 else "拿放 [小键1]"
        dash_lbl = "冲刺 [L]" if is_p1 else "冲刺 [小键3]"

        act_btns = [
            ('CHOP', chop_lbl, (255, 210, 0, 255)),
            ('PICK', pick_lbl, (0, 220, 255, 255)),
            ('DASH', dash_lbl, (255, 110, 50, 255))
        ]
        for act, lbl, glow_col in act_btns:
            active = act in actions
            if active:
                fill_col = glow_col
                outline_col = (255, 255, 255, 255)
                txt_col = (10, 10, 10)
            else:
                fill_col = (40, 45, 55, 180)
                outline_col = (70, 75, 90, 200)
                txt_col = (180, 180, 190)
            btn_w = 68
            draw.rounded_rectangle([(bx, by), (bx + btn_w, by + 26)], radius=6, fill=fill_col, outline=outline_col, width=1)
            draw.text((bx + 4, by + 4), lbl, font=font_btn, fill=txt_col)
            bx += btn_w + 6

        tilt_txt = f"左右:{player_obj.smooth_tilt_x:+.2f}"
        elev_txt = f"高低:{player_obj.smooth_rel_y:+.2f}"
        pitch_txt = f"前倾:{player_obj.smooth_pitch:+.2f}"
        hud_info = f"姿态 | {tilt_txt}  {elev_txt}  {pitch_txt}"
        draw.text((x0 + 12, y0 + 100), hud_info, font=font_sub, fill=(170, 190, 210))

    card_h = 122
    draw_player_card(8, 8, w // 2 - 6, card_h,
                     "◆ P1 蓝队大厨",
                     (0, 225, 255, 255),
                     (15, 25, 38, 225),
                     p1, is_p1=True)

    draw_player_card(w // 2 + 6, 8, w - 8, card_h,
                     "◆ P2 橙队大厨",
                     (255, 130, 20, 255),
                     (38, 24, 15, 225),
                     p2, is_p1=False)

    # 未校准时的常驻引导提示
    if not p1.is_calibrated and not p2.is_calibrated:
        draw.rounded_rectangle([(20, h // 2 - 50), (w - 20, h // 2 + 25)], radius=15,
                               fill=(15, 25, 40, 240), outline=(0, 220, 255, 255), width=2)
        draw.text((w // 2 - 125, h // 2 - 38), "★ 请就位并【双手叉腰】1秒", font=font_bold, fill=(0, 220, 255))
        draw.text((w // 2 - 145, h // 2 - 5), "系统将自动锁定当前中立站姿开玩！", font=font_hint, fill=(200, 220, 240))

    # 校准成功提示横幅
    if now < recalib_anim_until:
        draw.rounded_rectangle([(30, h // 2 - 60), (w - 30, h // 2 - 10)], radius=12,
                               fill=(0, 180, 120, 240), outline=(255, 255, 255, 255), width=2)
        draw.text((w // 2 - 110, h // 2 - 45), recalib_msg if recalib_msg else "★ 中立站姿已锁定！", font=font_bold, fill=(255, 255, 255))

    # 剧情翻页横幅
    if now < space_anim_until:
        draw.rounded_rectangle([(25, h // 2 - 80), (w - 25, h // 2 - 20)], radius=12,
                               fill=(255, 200, 0, 240), outline=(255, 255, 255, 255), width=2)
        draw.text((w // 2 - 120, h // 2 - 58), "★ 击掌翻页 [空格已触发]！", font=font_bold, fill=(20, 20, 20))

    # 底部体感操作速查指引卡
    guide_w = w - 16
    guide_x0 = 8
    guide_y0 = h - 66
    draw.rounded_rectangle([(guide_x0, guide_y0), (guide_x0 + guide_w, guide_y0 + 58)], radius=10,
                           fill=(18, 22, 28, 230), outline=(55, 70, 90, 200), width=1)
    
    line1 = f"FPS: {fps:.0f} | 踮脚=上↑ 深下蹲=下↓ 左倾=左← 右倾=右→ | 双手叉腰1秒校准"
    line2 = "【击掌拍手】剧情空格 | 【高抬单手】拿放 | 【深前倾】冲刺 | 【挥刀连击】切菜"
    draw.text((guide_x0 + 10, guide_y0 + 8), line1, font=font_hint, fill=(0, 255, 180))
    draw.text((guide_x0 + 10, guide_y0 + 32), line2, font=font_hint, fill=(220, 220, 230))

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def main():
    global recalib_anim_until, recalib_msg
    print("=" * 60)
    print("      Insight 9 ➔ Steam《胡闹厨房 2》双人体感控制器")
    print("     (姿态触发式校准：就位后【双手叉腰】1秒即可对齐开玩！)")
    print("=" * 60)

    _old_err = silence_stderr()
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_poses=2,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3
    )
    detector = vision.PoseLandmarker.create_from_options(options)
    restore_stderr(_old_err)

    print("\n【启动】正在开启 Insight 9 摄像头 (镜像模式)...")
    fast_cam = FastCamera(src=CAMERA_INDEX)
    time.sleep(0.3)

    controller = KeyController()
    p1 = PlayerTracker(player_id=1, keys=P1_KEYS, color=(0, 225, 255))
    p2 = PlayerTracker(player_id=2, keys=P2_KEYS, color=(255, 130, 20))

    win_id = "Overcooked2_Motion_Dashboard"
    display_title = "胡闹厨房 2 - 赛博体感追踪仪表盘"
    
    DISP_H = 880
    DISP_W = int(DISP_H * (1088.0 / 1920.0))

    cv2.namedWindow(win_id, cv2.WINDOW_NORMAL)
    # 立即渲染一帧底色迫使 Windows 创建真实 HWND，杜绝 320x320 缩水小方块
    blank_init = np.zeros((DISP_H, DISP_W, 3), dtype=np.uint8)
    cv2.imshow(win_id, blank_init)
    cv2.waitKey(1)

    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)

    hwnd = user32.FindWindowW(None, win_id)
    if hwnd:
        user32.SetWindowTextW(hwnd, display_title)
        user32.SetWindowPos(hwnd, -2, 60, 40, DISP_W, DISP_H, 0x0040)

    is_pip_mode = False
    last_f9_time = 0.0
    last_c_time = 0.0

    def apply_window_mode(pip_on):
        h = user32.FindWindowW(None, display_title)
        if not h:
            h = user32.FindWindowW(None, win_id)
        if not h:
            return
        if pip_on:
            w_pip = 270
            h_pip = 475
            x_pip = max(10, screen_w - w_pip - 20)
            y_pip = 25
            user32.SetWindowPos(h, -1, x_pip, y_pip, w_pip, h_pip, 0x0010 | 0x0040)
            print("【模式】已切换为：单屏画中画置顶小窗！")
        else:
            w_big = DISP_W
            h_big = DISP_H
            user32.SetWindowPos(h, -2, 60, 40, w_big, h_big, 0x0010 | 0x0040)
            print("【模式】已切换为：双屏高清大窗模式！")

    prev_time = time.time()
    fps_smooth = 60.0

    print("【就绪】姿态触发式校准引擎已启动！")
    print("  * 彻底告别赶时间：启动后您可以从容走位，就位后【双手叉腰】1秒即可锁定站姿！")
    print("  * 上下走位规则：自然微屈膝【下蹲】=向下走[S]，站直向上【踮脚/拔高】=向上走[W]！")
    print("  * 左右走位规则：身体向左侧倾=向左走[A]，身体向右侧倾=向右走[D]！")
    print("  * 游戏中随时按【C】也可秒级重置对齐！\n")

    try:
        title_set = False
        while True:
            now = time.time()
            update_scheduled_keys(now)

            if cv2.getWindowProperty(win_id, cv2.WND_PROP_VISIBLE) < 1:
                break

            if not title_set:
                h = user32.FindWindowW(None, win_id)
                if h:
                    user32.SetWindowTextW(h, display_title)
                    user32.SetWindowPos(h, -2, 60, 40, DISP_W, DISP_H, 0x0040)
                    title_set = True

            # 随时按 [C] 重新秒校准当前站位
            if is_global_key_pressed(VK_C):
                if now - last_c_time > 0.50:
                    if 'p1_pose' in locals() and p1_pose: p1.lock_baseline(p1_pose)
                    if 'p2_pose' in locals() and p2_pose: p2.lock_baseline(p2_pose)
                    recalib_msg = "🎯 键盘 [C] 秒校准完成！"
                    recalib_anim_until = now + 1.2
                    last_c_time = now
                    print("【秒校准】站姿已瞬间重置居中！")

            if is_global_key_pressed(VK_F9):
                if now - last_f9_time > 0.45:
                    is_pip_mode = not is_pip_mode
                    apply_window_mode(is_pip_mode)
                    last_f9_time = now

            ret, frame = fast_cam.read_latest()
            if not ret or frame is None:
                time.sleep(0.002)
                continue

            instant_fps = 1.0 / max(0.001, now - prev_time)
            fps_smooth = fps_smooth * 0.9 + instant_fps * 0.1
            prev_time = now

            ai_input = cv2.resize(frame, (272, 480))
            rgb_small = cv2.cvtColor(ai_input, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_small)
            result = detector.detect(mp_image)

            poses = result.pose_landmarks

            p1_pose = None
            p2_pose = None

            if len(poses) == 1:
                cx = (poses[0][11].x + poses[0][12].x) / 2.0
                if cx < 0.5:
                    p1_pose = poses[0]
                else:
                    p2_pose = poses[0]
            elif len(poses) >= 2:
                c0_x = (poses[0][11].x + poses[0][12].x) / 2.0
                c1_x = (poses[1][11].x + poses[1][12].x) / 2.0
                if c0_x < c1_x:
                    p1_pose, p2_pose = poses[0], poses[1]
                else:
                    p1_pose, p2_pose = poses[1], poses[0]

            p1.update(p1_pose, controller, now)
            p2.update(p2_pose, controller, now)

            display_img = cv2.resize(frame, (DISP_W, DISP_H), interpolation=cv2.INTER_AREA)

            draw_cyber_skeleton(display_img, p1_pose, (255, 225, 0), DISP_W, DISP_H)
            draw_cyber_skeleton(display_img, p2_pose, (20, 130, 255), DISP_W, DISP_H)

            for p, ring_col in [(p1, (255, 225, 0)), (p2, (20, 130, 255))]:
                if len(poses) > 0:
                    pose = p1_pose if p.player_id == 1 else p2_pose
                    if pose:
                        nose = pose[0]
                        sm_x = (pose[11].x + pose[12].x) / 2.0
                        sm_y = (pose[11].y + pose[12].y) / 2.0
                        
                        nx, ny = int(nose.x * DISP_W), int(nose.y * DISP_H)
                        sx, sy = int(sm_x * DISP_W), int(sm_y * DISP_H)
                        
                        is_active = len(p.active_actions) > 0
                        line_col = (0, 255, 120) if not is_active else (0, 70, 255)
                        cv2.line(display_img, (sx, sy), (nx, ny), line_col, 2, cv2.LINE_AA)
                        cv2.circle(display_img, (sx, sy), 4, ring_col, -1, cv2.LINE_AA)

            display_img = draw_cyber_hud(display_img, p1, p2, fps_smooth, now)
            cv2.imshow(win_id, display_img)

            cv_key = cv2.waitKey(1) & 0xFF
            if (is_global_key_pressed(VK_CONTROL) and is_global_key_pressed(VK_Q)) or is_global_key_pressed(VK_F10):
                print("【退出】收到管理员退出指令。")
                break

    finally:
        controller.release_all()
        fast_cam.release()
        cv2.destroyAllWindows()
        print("【安全退出】所有按键与资源已释放。")

if __name__ == '__main__':
    main()
