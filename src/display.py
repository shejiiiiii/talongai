# src/display.py
"""
Pygame dashboard — runs as its own process.

Entry point: run_display(shared_state)
Called by main.py via multiprocessing.Process.

The shared_state is a multiprocessing.Manager proxy dict with keys:
    running   : bool  — set False to shut down this process
    phase     : str   — current phase label shown in the header
    status_0  : str   — status for Eggplant #1 panel
    status_1  : str   — status for Eggplant #2 panel
"""

import os
import math
import time
import threading


# ── Resolution — change to match your monitor ─────────────────────────────────
W, H       = 1024, 600
FULLSCREEN = True
FPS        = 60

# ── Palette ───────────────────────────────────────────────────────────────────
C = {
    "bg"          : (8,   7,   14 ),
    "panel"       : (16,  14,  26 ),
    "panel_border": (55,  30,  80 ),
    "ep"          : (108, 32,  130),
    "ep_dark"     : (60,  10,  80 ),
    "ep_lite"     : (165, 75,  200),
    "stem"        : (55,  88,  28 ),
    "leaf"        : (72,  130, 42 ),
    "worm"        : (205, 165, 118),
    "worm_dk"     : (155, 108, 68 ),
    "white"       : (255, 255, 255),
    "gray"        : (110, 105, 135),
    "gray_dk"     : (28,  25,  42 ),
    "gray_md"     : (45,  40,  62 ),
    "s0"          : (170, 85,  220),
    "s1"          : (75,  205, 175),
    "healthy"     : (72,  220, 115),
    "infested"    : (255, 68,  68 ),
    "warn"        : (255, 178, 48 ),
    "boot_bg"     : (5,   4,   10 ),
}

BOOT_TOTAL  = 5.0
FADEIN_DUR  = 1.0
FADEOUT_DUR = 0.7

MACRO_KEYS = [
    ("1", "Turn OFF the device"),
    ("2", "SCAN again"),
    ("3", "STOP scanning"),
    ("4", "RESET the device"),
]

BOOT_MESSAGES = [
    (0.10, "Initializing display driver..."),
    (0.22, "Mounting I2C bus..."),
    (0.34, "Connecting to MUX (0x70)..."),
    (0.46, "Spectral Sensor 0  ->  OK"),
    (0.57, "Spectral Sensor 1  ->  OK"),
    (0.66, "Motor driver  ->  ENABLED"),
    (0.75, "Loading TensorFlow runtime..."),
    (0.85, "Loading CNN model weights..."),
    (0.93, "Sensor warm-up in progress..."),
    (1.00, "System ready."),
]


# ─────────────────────────────────────────────────────────────────────────────
#  UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def lerp(a, b, t):     return a + (b - a) * t
def lerp_col(a, b, t): return tuple(int(lerp(a[i], b[i], t)) for i in range(3))
def ease_out(t):       return 1 - (1 - t) ** 3
def ease_in_out(t):    return t * t * (3 - 2 * t)

def rrect(surf, color, rect, r, width=0, alpha=None):
    import pygame
    rx  = pygame.Rect(rect)
    rgb = color[:3]
    a   = alpha if alpha is not None else (color[3] if len(color) == 4 else None)
    if a is not None:
        tmp = pygame.Surface((rx.w, rx.h), pygame.SRCALPHA)
        pygame.draw.rect(tmp, (*rgb, a), (0, 0, rx.w, rx.h), width,
                         border_radius=r)
        surf.blit(tmp, rx.topleft)
    else:
        pygame.draw.rect(surf, rgb, rx, width, border_radius=r)


# ─────────────────────────────────────────────────────────────────────────────
#  EGGPLANT + WORM LOGO
# ─────────────────────────────────────────────────────────────────────────────

def draw_logo(target, cx, cy, s=1.0, tint=None):
    import pygame
    bw, bh = int(70*s), int(115*s)

    for i in range(6):
        t   = i / 5
        col = lerp_col(C["ep_dark"], C["ep_lite"], t * 0.55)
        ew  = max(4, bw - i*int(10*s))
        eh  = max(4, bh - i*int(8*s))
        ex  = cx - ew//2 - int(4*s*t)
        ey  = cy - eh//2 + int(8*s)
        pygame.draw.ellipse(target, col, (ex, ey, ew, eh))

    if tint:
        ow, oh = bw + int(10*s), bh + int(10*s)
        ts2    = pygame.Surface((ow, oh), pygame.SRCALPHA)
        pygame.draw.ellipse(ts2, (*tint, 90), (0, 0, ow, oh))
        target.blit(ts2, (cx - ow//2, cy - oh//2 + int(8*s)))

    hlw = max(2, int(9*s)); hlh = max(2, int(40*s))
    hl  = pygame.Surface((hlw, hlh), pygame.SRCALPHA)
    pygame.draw.ellipse(hl, (255,255,255,55), (0, 0, hlw, hlh))
    target.blit(hl, (cx - int(22*s), cy - int(24*s)))

    sx   = cx - int(2*s)
    sbot = cy - bh//2 + int(8*s) + int(6*s)
    stop = cy - bh//2 + int(8*s) - int(22*s)
    pygame.draw.line(target, C["stem"], (sx, sbot),
                     (sx + int(3*s), stop), max(1, int(6*s)))

    lcx, lcy = cx, cy - bh//2 + int(8*s)
    for deg, length in [(-42,22),(0,17),(42,20),(-72,14),(72,14)]:
        a  = math.radians(deg - 90)
        lx = lcx + int(math.cos(a)*length*s)
        ly = lcy + int(math.sin(a)*length*s)
        pygame.draw.line(target, C["leaf"], (lcx, lcy), (lx, ly),
                         max(1, int(4*s)))
        pygame.draw.circle(target, C["leaf"], (lx, ly), max(1, int(3*s)))

    wx, wy = cx + int(25*s), cy - int(10*s)
    segs   = 6
    for i in range(segs):
        t    = i / (segs - 1)
        segx = wx + int(math.cos(t*0.9)*28*s*t)
        segy = wy - int(math.sin(t*1.2)*18*s*t)
        r    = max(2, int((8 - i*0.8)*s))
        pygame.draw.circle(target, C["worm_dk"], (segx+1, segy+2), r)
        pygame.draw.circle(target, C["worm"],    (segx,   segy  ), r)
        if i < segs - 1:
            nx = wx + int(math.cos((i+1)/(segs-1)*0.9)*28*s*(i+1)/(segs-1))
            ny = wy - int(math.sin((i+1)/(segs-1)*1.2)*18*s*(i+1)/(segs-1))
            pygame.draw.line(target, C["worm_dk"],
                             (segx, segy), (nx, ny), max(1, int(3*s)))

    hx = wx + int(math.cos(0.9)*28*s)
    hy = wy - int(math.sin(1.2)*18*s)
    hr = max(3, int(9*s))
    pygame.draw.circle(target, C["worm_dk"], (hx+1, hy+2), hr)
    pygame.draw.circle(target, C["worm"],    (hx,   hy  ), hr)
    er = max(1, int(2*s))
    pygame.draw.circle(target, (28,18,8),
                       (hx - int(3*s), hy - int(3*s)), er)
    pygame.draw.circle(target, C["white"],
                       (hx - int(3*s)-1, hy - int(3*s)-1), max(1, er-1))
    pygame.draw.arc(target, (75,35,8),
                    (hx - int(4*s), hy, int(7*s), int(4*s)),
                    math.pi, 0, max(1, int(2*s)))


# ─────────────────────────────────────────────────────────────────────────────
#  STATUS METADATA
# ─────────────────────────────────────────────────────────────────────────────

STATUS_MAP = {
    "INFESTED"   : {"col": C["infested"], "label": "INFESTED",  "tint": (255, 50, 50)},
    "HEALTHY"    : {"col": C["healthy"],  "label": "HEALTHY",   "tint": None},
    "SCANNING"   : {"col": C["warn"],     "label": "SCANNING",  "tint": None},
    "PROCESSING" : {"col": C["warn"],     "label": "PROCESSING","tint": None},
    "LOADING"    : {"col": C["s0"],       "label": "LOADING",   "tint": None},
    "READY"      : {"col": C["healthy"],  "label": "READY",   "tint": None},
    "READY"      : {"col": C["healthy"],  "label": "READY",   "tint": None},
    "WAITING"    : {"col": C["gray"],     "label": "WAITING",   "tint": None},
    "MOVING"     : {"col": C["gray"],     "label": "MOVING",    "tint": None},
    "RETURNING"  : {"col": C["gray"],     "label": "RETURNING", "tint": None},
    "HOMING"     : {"col": C["gray"],     "label": "HOMING",    "tint": None},
    "ERROR"      : {"col": C["infested"], "label": "ERROR",     "tint": (255, 50, 50)},
    "ERROR"       : {"col": C["infested"], "label": "ERROR",     "tint": (255, 50, 50)},
}

def _meta(status: str) -> dict:
    s = status.upper()
    for key, val in STATUS_MAP.items():
        if key in s:
            return val
    return {"col": C["gray"], "label": s[:14], "tint": None}


# ─────────────────────────────────────────────────────────────────────────────
#  MACRO KEY BAR
# ─────────────────────────────────────────────────────────────────────────────

def draw_macro_bar(surf, rect, tick, accent, fonts):
    import pygame
    x, y, w, h = rect
    row_h = h // 4
    lx    = x + 8
    for i, (num, action) in enumerate(MACRO_KEYS):
        line_s = fonts["tiny"].render(f"{num} - {action}...", True, C["gray"])
        surf.blit(line_s, (lx, y + i*row_h + (row_h - line_s.get_height())//2))

    period   = 10.0
    cycle    = (tick % period) / period
    ping     = 1.0 - abs(cycle * 2 - 1)
    bounce_i = min(3, max(0, int(ping * 3 + 0.5)))

    key_sz  = min(38, h - 6)
    key_gap = 8
    keys_w  = 4*key_sz + 3*key_gap
    kx0     = x + w - keys_w - 6
    ky      = y + (h - key_sz) // 2

    for i, (num, _) in enumerate(MACRO_KEYS):
        kx = kx0 + i*(key_sz + key_gap)
        is_active = (i == bounce_i)
        if is_active:
            pulse    = 0.7 + 0.3*math.sin(tick*6)
            glow_col = lerp_col(accent, C["white"], pulse*0.35)
            rrect(surf, (*glow_col, 35),
                  (kx-4, ky-4, key_sz+8, key_sz+8), 10, alpha=35)
            rrect(surf, glow_col, (kx, ky, key_sz, key_sz), 8)
            txt_col = C["white"]
        else:
            rrect(surf, C["gray_md"], (kx, ky, key_sz, key_sz), 8)
            rrect(surf, C["gray"],    (kx, ky, key_sz, key_sz), 8, width=1)
            txt_col = C["gray"]
        n_s = fonts["key"].render(num, True, txt_col)
        surf.blit(n_s, (kx + key_sz//2 - n_s.get_width()//2,
                        ky + key_sz//2 - n_s.get_height()//2))


# ─────────────────────────────────────────────────────────────────────────────
#  SENSOR PANEL
# ─────────────────────────────────────────────────────────────────────────────

def draw_sensor_panel(surf, rect, name, accent, phase, status, fonts, tick):
    import pygame
    x, y, w, h = rect
    meta = _meta(status)
    col  = meta["col"]
    tint = meta["tint"]

    rrect(surf, C["panel"], rect, 18)
    rrect(surf, C["panel_border"], rect, 18, width=1)
    if tint:
        rrect(surf, (*tint, 28), rect, 18, alpha=28)

    gw = w - 40
    gs = pygame.Surface((gw, 3), pygame.SRCALPHA)
    for xi in range(gw):
        a = int(200 * math.sin(math.pi * xi / gw))
        pygame.draw.line(gs, (*accent, a), (xi,0),(xi,2))
    surf.blit(gs, (x+20, y+2))

    HDR_H   = 46
    MACRO_H = max(72, min(96, int(h*0.17)))
    inner_h = h - HDR_H - MACRO_H - 4

    logo_zone_h   = int(inner_h * 0.56)
    status_zone_h = inner_h - logo_zone_h

    logo_zone_top   = y + HDR_H + 4
    status_zone_top = logo_zone_top + logo_zone_h
    macro_zone_top  = y + h - MACRO_H

    # Header
    name_s = fonts["panel_title"].render(name, True, C["white"])
    surf.blit(name_s, (x+18, y + (HDR_H - name_s.get_height())//2))

    ph_s = fonts["tiny"].render(phase.upper()[:14], True, C["gray"])
    ph_w = ph_s.get_width() + 16
    ph_x = x + w - ph_w - 14
    ph_y = y + (HDR_H - ph_s.get_height() - 8)//2
    rrect(surf, C["gray_dk"], (ph_x, ph_y, ph_w, ph_s.get_height()+8), 6)
    rrect(surf, C["gray_md"], (ph_x, ph_y, ph_w, ph_s.get_height()+8), 6, width=1)
    surf.blit(ph_s, (ph_x+8, ph_y+4))
    pygame.draw.line(surf, C["gray_dk"],
                     (x+14, y+HDR_H-1), (x+w-14, y+HDR_H-1), 1)

    # Logo
    logo_s  = min(1.8, (logo_zone_h - 10) / 135.0)
    logo_sw = int(90 * logo_s) + 20
    logo_sh = int(135 * logo_s) + 10
    logo_sf = pygame.Surface((logo_sw, logo_sh), pygame.SRCALPHA)
    draw_logo(logo_sf, logo_sw//2, logo_sh//2 + 4, s=logo_s, tint=tint)
    lx = x + w//2 - logo_sw//2
    ly = logo_zone_top + (logo_zone_h - logo_sh)//2
    surf.blit(logo_sf, (lx, ly))

    # Status box
    sb_pad = 12
    sb_x   = x + sb_pad
    sb_w   = w - sb_pad*2
    sb_h   = status_zone_h - 10
    sb_y   = status_zone_top + 5

    if "SCANNING" in status.upper() or "PROCESSING" in status.upper():
        box_bg = (42, 30, 5)
    elif tint:
        box_bg = tuple(max(0, v//5) for v in tint)
    else:
        box_bg = (18, 16, 28)

    rrect(surf, (*box_bg, 210), (sb_x, sb_y, sb_w, sb_h), 12, alpha=210)

    if "SCANNING" in status.upper() or "PROCESSING" in status.upper():
        pulse = 0.6 + 0.4*math.sin(tick*4)
        rrect(surf, (*col, int(pulse*32)),
              (sb_x, sb_y, sb_w, sb_h), 12, alpha=int(pulse*32))

    label = meta["label"]
    st_s  = fonts["status"].render(label, True, col)
    surf.blit(st_s, (sb_x + sb_w//2 - st_s.get_width()//2,
                     sb_y + sb_h//2 - st_s.get_height()//2))

    ul_w = min(st_s.get_width()+24, sb_w-20)
    ul_y = sb_y + sb_h - 7
    for xi in range(ul_w):
        t = xi / ul_w
        a = int(200 * math.sin(math.pi*t))
        pygame.draw.line(surf, (*col, a),
                         (sb_x + sb_w//2 - ul_w//2 + xi, ul_y),
                         (sb_x + sb_w//2 - ul_w//2 + xi, ul_y+2))

    pygame.draw.line(surf, C["gray_dk"],
                     (x+14, macro_zone_top-1), (x+w-14, macro_zone_top-1), 1)
    draw_macro_bar(surf, (x+10, macro_zone_top+4, w-20, MACRO_H-6),
                   tick, accent, fonts)


# ─────────────────────────────────────────────────────────────────────────────
#  BOOT SCREEN
# ─────────────────────────────────────────────────────────────────────────────

class BootScreen:
    def __init__(self, screen, fonts):
        self.screen = screen
        self.fonts  = fonts
        self.start  = time.time()
        self.done   = False
        self._fo    = False
        self._fo_t  = 0.0

        logo_h   = int(H * 0.42)
        logo_w   = int(logo_h * 0.92)
        self._ls = logo_h / 175.0
        self.logo = _make_logo_surface(logo_w, logo_h, self._ls)

    @property
    def elapsed(self):  return time.time() - self.start
    @property
    def progress(self): return min(1.0, self.elapsed / BOOT_TOTAL)

    @property
    def content_alpha(self):
        return int(ease_out(min(1.0, self.elapsed / FADEIN_DUR)) * 255)

    @property
    def overlay_alpha(self):
        if not self._fo: return 0
        t = min(1.0, (time.time() - self._fo_t) / FADEOUT_DUR)
        return int(ease_in_out(t) * 255)

    def update(self):
        if self.progress >= 1.0 and not self._fo:
            self._fo   = True
            self._fo_t = time.time()
        if self._fo and (time.time() - self._fo_t) >= FADEOUT_DUR:
            self.done = True

    def draw(self):
        import pygame
        self.screen.fill(C["boot_bg"])
        ca = self.content_alpha
        cx = W // 2

        title_y = int(H * 0.04)
        sub_y   = int(H * 0.130)
        sep_y   = int(H * 0.170)
        logo_y  = int(H * 0.175)
        bar_y   = int(H * 0.785)
        msg_y   = bar_y + int(H * 0.028)
        past_y  = msg_y  + int(H * 0.032)
        ver_y   = H - int(H * 0.042)

        title_s = self.fonts["boot_title"].render("TALONG  AI", True,
                    lerp_col(C["ep_lite"], C["white"], 0.55))
        title_s.set_alpha(ca)
        self.screen.blit(title_s, (cx - title_s.get_width()//2, title_y))

        sub_s = self.fonts["boot_sub"].render(
            "Eggplant Pest Detection System", True, C["gray"])
        sub_s.set_alpha(int(ca * 0.7))
        self.screen.blit(sub_s, (cx - sub_s.get_width()//2, sub_y))

        sep_w    = int(W * 0.27)
        sep_surf = pygame.Surface((sep_w, 1), pygame.SRCALPHA)
        for xi in range(sep_w):
            t = xi / sep_w
            a = int(ca * 0.55 * math.sin(math.pi * t))
            pygame.draw.line(sep_surf, (*C["ep_lite"], a), (xi,0),(xi,0))
        self.screen.blit(sep_surf, (cx - sep_w//2, sep_y))

        logo = self.logo.copy()
        logo.set_alpha(ca)
        self.screen.blit(logo, (cx - logo.get_width()//2, logo_y))

        bar_w = int(W * 0.34)
        bar_h = int(H * 0.009)
        bar_x = cx - bar_w//2
        rrect(self.screen, (*C["gray_dk"], int(ca*0.7)),
              (bar_x, bar_y, bar_w, bar_h), 3, alpha=int(ca*0.7))
        fw = max(0, int(bar_w * self.progress))
        if fw > 0:
            for xi in range(fw):
                t = xi / max(1, bar_w)
                col = lerp_col(C["ep"], C["ep_lite"], t)
                pygame.draw.line(self.screen, col,
                                 (bar_x+xi, bar_y), (bar_x+xi, bar_y+bar_h-1))

        current_msg = ""
        for thresh, msg in BOOT_MESSAGES:
            if self.progress >= thresh:
                current_msg = msg
        if current_msg:
            msg_s = self.fonts["boot_log"].render("->  " + current_msg, True, C["ep_lite"])
            msg_s.set_alpha(ca)
            self.screen.blit(msg_s, (cx - msg_s.get_width()//2, msg_y))

        done_msgs = [(t, m) for t, m in BOOT_MESSAGES
                     if self.progress > t + 0.02 and m != current_msg]
        row_h = int(H * 0.030)
        for idx, (_, m) in enumerate(reversed(done_msgs[-2:])):
            fade = int(ca * (0.27 - idx*0.11))
            if fade <= 0: continue
            pm_s = self.fonts["tiny"].render("    " + m, True, C["gray"])
            pm_s.set_alpha(fade)
            self.screen.blit(pm_s, (cx - pm_s.get_width()//2, past_y + idx*row_h))

        ver_s = self.fonts["tiny"].render("v3.0  |  SE-CNN", True, C["gray_dk"])
        ver_s.set_alpha(int(ca * 0.5))
        self.screen.blit(ver_s, (cx - ver_s.get_width()//2, ver_y))

        oa = self.overlay_alpha
        if oa > 0:
            ov = pygame.Surface((W, H))
            ov.fill(C["boot_bg"])
            ov.set_alpha(oa)
            self.screen.blit(ov, (0, 0))


# ─────────────────────────────────────────────────────────────────────────────
#  DASHBOARD SCREEN
# ─────────────────────────────────────────────────────────────────────────────

def _make_logo_surface(logo_w, logo_h, s):
    import pygame
    surf = pygame.Surface((logo_w + 20, logo_h + 10), pygame.SRCALPHA)
    draw_logo(surf, (logo_w + 20)//2, (logo_h + 10)//2 + 10, s=s)
    return surf


class DashScreen:
    def __init__(self, screen, fonts, state):
        import pygame
        self.screen  = screen
        self.fonts   = fonts
        self.state   = state
        self.tick    = 0.0
        self.last    = time.time()
        self.alpha   = 0
        self.fi_t    = time.time()
        self.logo_sm = pygame.Surface((44, 50), pygame.SRCALPHA)
        draw_logo(self.logo_sm, 22, 28, s=0.35)

    def update(self):
        now = time.time()
        dt  = now - self.last
        self.last  = now
        self.tick += dt
        fi = min(1.0, (now - self.fi_t) / 0.9)
        self.alpha = int(ease_out(fi) * 255)

    def draw(self):
        import pygame
        self.screen.fill(C["bg"])

        for gx in range(0, W, 28):
            for gy in range(58, H, 28):
                a = int(12 + 5*math.sin(self.tick*0.6 + gx*0.04 + gy*0.04))
                pygame.draw.circle(self.screen, (*C["ep_dark"], a), (gx, gy), 1)

        bar_h = 50
        rrect(self.screen, C["panel"], (0, 0, W, bar_h), 0)
        pygame.draw.line(self.screen, C["panel_border"],
                         (0, bar_h), (W, bar_h), 1)

        self.screen.blit(self.logo_sm, (10, 2))

        title_s = self.fonts["bar_title"].render(
            "TALONG AI  -  PEST DETECTION", True, C["white"])
        self.screen.blit(title_s, (62, 14))

        ts_s = self.fonts["sub"].render(time.strftime("%H:%M:%S"), True, C["gray"])
        self.screen.blit(ts_s, (W - ts_s.get_width() - 16, 16))

        pulse = 0.5 + 0.5*math.sin(self.tick*2.5)
        for xi in range(W):
            t   = xi / W
            col = lerp_col(lerp_col(C["ep"], C["s0"], t),
                           lerp_col(C["ep"], C["s1"], t), pulse*0.12)
            pygame.draw.line(self.screen, col, (xi, bar_h-2), (xi, bar_h-1))

        mg  = 12
        top = bar_h + mg
        pw  = (W - mg*3) // 2
        ph  = H - top - mg

        phase   = self.state.get("phase",    "")
        status0 = self.state.get("status_0", "SYSTEM READY")
        status1 = self.state.get("status_1", "SYSTEM READY")

        for rect, name, accent, status in [
            ((mg,      top, pw, ph), "EGGPLANT  #1", C["s0"], status0),
            ((mg*2+pw, top, pw, ph), "EGGPLANT  #2", C["s1"], status1),
        ]:
            draw_sensor_panel(self.screen, rect, name, accent,
                              phase, status, self.fonts, self.tick)

        if self.alpha < 255:
            ov = pygame.Surface((W, H))
            ov.fill(C["bg"])
            ov.set_alpha(255 - self.alpha)
            self.screen.blit(ov, (0, 0))


# ─────────────────────────────────────────────────────────────────────────────
#  RENDER LOOP (internal)
# ─────────────────────────────────────────────────────────────────────────────

def _render_loop(state, screen=None, queue=None):
    """
    queue: optional multiprocessing.Queue
        When provided (spawned-process mode), each frame drains pending
        state updates from the main process before drawing.
    """
    import pygame

    if screen is None:
        try:
            flags = pygame.FULLSCREEN if FULLSCREEN else 0
            screen = pygame.display.set_mode((W, H), flags)
        except Exception as e:
            print(f"[Display] pygame.display.set_mode failed: {e}")
            state["running"] = False
            return

    pygame.display.set_caption("Talong AI")
    pygame.mouse.set_visible(False)
    clock = pygame.time.Clock()

    def fnt(size, bold=False):
        for name in ["Courier New","Consolas","DejaVu Sans Mono",
                     "Lucida Console","monospace"]:
            try:
                return pygame.font.SysFont(name, size, bold=bold)
            except Exception:
                pass
        return pygame.font.Font(None, size)

    fs = W / 800.0
    fonts = {
        "boot_title" : fnt(int(46*fs), bold=True),
        "boot_sub"   : fnt(int(16*fs)),
        "boot_log"   : fnt(int(15*fs)),
        "bar_title"  : fnt(int(20*fs), bold=True),
        "panel_title": fnt(int(20*fs), bold=True),
        "status"     : fnt(int(30*fs), bold=True),
        "sub"        : fnt(int(16*fs)),
        "tiny"       : fnt(int(14*fs)),
        "key"        : fnt(int(17*fs), bold=True),
    }

    boot      = BootScreen(screen, fonts)
    dash      = None
    scr_state = "boot"

    while state.get("running", True):
        # Drain pending state updates from the main process
        if queue is not None:
            try:
                while True:
                    update = queue.get_nowait()
                    state.update(update)
            except Exception:
                pass

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                state["running"] = False
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    state["running"] = False
                if ev.key == pygame.K_SPACE and scr_state == "boot":
                    boot._fo   = True
                    boot._fo_t = time.time()

        if scr_state == "boot":
            boot.update()
            boot.draw()
            if boot.done:
                scr_state = "dash"
                dash = DashScreen(screen, fonts, state)
        else:
            dash.update()
            dash.draw()

        pygame.display.flip()
        clock.tick(FPS)

    screen.fill((0, 0, 0))
    pygame.display.flip()
    pygame.quit()


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINT  —  called from the main thread in main.py
# ─────────────────────────────────────────────────────────────────────────────

def run_display(state: dict, screen=None):
    """
    Run the render loop on the calling (main) thread.

    Args:
        state:  shared dict from main.py
        screen: pre-created pygame.Surface from main.py.
                If None, this function will call pygame.init() and set_mode()
                itself — but on ARM/Pi this will crash if TF was loaded first.
                Always pass the pre-created screen from main.py.
    """
    import traceback

    try:
        _render_loop(state, screen=screen)
    except Exception:
        print("[Display] Fatal error:")
        traceback.print_exc()
