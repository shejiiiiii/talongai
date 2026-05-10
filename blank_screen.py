# blank_screen.py
"""
Standby black screen.

Launched by macro_listener.py when the device power state is 'off'.
Displays a plain black screen and waits — the user pressing button 1
will kill this and launch main.py instead.

Run: python3 blank_screen.py
"""

import os
os.environ.setdefault("DISPLAY", ":0")
os.environ["SDL_VIDEODRIVER"]       = "x11"
os.environ["SDL_RENDERDRIVER"]      = "software"
os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"

import pygame
import sys

W, H       = 1024, 600
FULLSCREEN = True


def main():
    pygame.init()

    try:
        flags  = pygame.FULLSCREEN if FULLSCREEN else 0
        screen = pygame.display.set_mode((W, H), flags)
    except Exception:
        screen = pygame.display.set_mode((W, H))

    pygame.display.set_caption("Standby")
    pygame.mouse.set_visible(False)

    clock = pygame.time.Clock()

    while True:
        for event in pygame.event.get():
            # Allow ESC to exit during development/testing
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)

        screen.fill((0, 0, 0))   # pure black
        pygame.display.flip()
        clock.tick(10)            # low FPS — nothing to animate


if __name__ == "__main__":
    main()
