import time
import pygame
import numpy as np

h = 1         # spatial step width
k = 1         # time step width
dimx = 400    # width of the simulation domain
dimy = 400    # height of the simulation domain
cellsize = 2  # display size of a cell in pixel


def init_simulation(option):
    global kappa
    u = np.zeros((3, dimx, dimy))           # The three dimensional simulation grid
    c = 0.5                                # The "original" wave propagation speed
    alpha = np.zeros((dimx, dimy))          # wave propagation velocities of the entire simulation domain
    alpha[0:dimx, 0:dimy] = ((c*k) / h)**2  # will be set to a constant value of tau
    kappa = 1 * alpha / 1

    if option == 3 or option == 4:
        alpha[0:196, dimy - 200] = 0
        alpha[204:400, dimy - 200] = 0

    if option == 4:
        alpha[0:128, dimy - 100] = 0
        alpha[136:264, dimy - 100] = 0
        alpha[272:400, dimy - 100] = 0

    return u, alpha


def update(u, alpha):
    u[2] = u[1]
    u[1] = u[0]

    # This switch is for educational purposes. The fist implementation is approx 50 times slower in python!
    use_terribly_slow_implementation = False
    if use_terribly_slow_implementation:
        # Version 1: Easy to understand but terribly slow!
        for c in range(1, dimx-1):
            for r in range(1, dimy-1):
                u[0, c, r] = alpha[c, r] * (u[1, c-1, r] + u[1, c+1, r] + u[1, c, r-1] + u[1, c, r+1] - 4*u[1, c, r])
                u[0, c, r] += 2 * u[1, c, r] - u[2, c, r]
    else:

        u[0, 1:dimx-1, 1:dimy-1] = alpha[1:dimx-1, 1:dimy-1] * (u[1, 0:dimx-2, 1:dimy-1] +
                                            u[1, 2:dimx,   1:dimy-1] +
                                            u[1, 1:dimx-1, 0:dimy-2] +
                                            u[1, 1:dimx-1, 2:dimy] - 4*u[1, 1:dimx-1, 1:dimy-1]) \
                                        + 2 * u[1, 1:dimx-1, 1:dimy-1] - u[2, 1:dimx-1, 1:dimy-1]


    u[0, 1:dimx-1, 1:dimy-1] *= 0.998

    boundary_size = 1
    mur = True
    if mur:
        update_boundary(u, boundary_size)


def update_boundary(u, sz) -> None:
    c = dimx - 1
    damping = 0.5

    u[0, dimx - sz - 1:c, 1:dimy - 1] = u[1, dimx - sz - 2:c - 1, 1:dimy - 1] \
                                        + (kappa[dimx - sz - 1:c, 1:dimy - 1] - 1) \
                                        / (kappa[dimx - sz - 1:c, 1:dimy - 1] + 1) \
                                        * (u[0, dimx - sz - 2:c - 1, 1:dimy - 1] - u[1, dimx - sz - 1:c,
                                                                                   1:dimy - 1]) * damping

    c = 0
    u[0, c:sz, 1:dimy - 1] = u[1, c + 1:sz + 1, 1:dimy - 1] \
                             + (kappa[c:sz, 1:dimy - 1] - 1) \
                             / (kappa[c:sz, 1:dimy - 1] + 1) \
                             * (u[0, c + 1:sz + 1, 1:dimy - 1] - u[1, c:sz, 1:dimy - 1]) * damping

    r = dimy - 1
    u[0, 1:dimx - 1, dimy - 1 - sz:r] = u[1, 1:dimx - 1, dimy - 2 - sz:r - 1] \
                                        + (kappa[1:dimx - 1, dimy - 1 - sz:r] - 1) \
                                        / (kappa[1:dimx - 1, dimy - 1 - sz:r] + 1) \
                                        * (u[0, 1:dimx - 1, dimy - 2 - sz:r - 1] - u[1, 1:dimx - 1,
                                                                                   dimy - 1 - sz:r]) * damping

    r = 0
    u[0, 1:dimx - 1, r:sz] = u[1, 1:dimx - 1, r + 1:sz + 1] \
                             + (kappa[1:dimx - 1, r:sz] - 1) \
                             / (kappa[1:dimx - 1, r:sz] + 1) \
                             * (u[0, 1:dimx - 1, r + 1:sz + 1] - u[1, 1:dimx - 1, r:sz]) * damping


def place_raindrops(u, pos):
    x = int(pos[0] / 2)
    y = int(pos[1] / 2)
    u[0, x-2:x+2, y-2:y+2] = 250


def main(option):
    tick = 0
    start_time = time.time()

    pygame.init()
    display = pygame.display.set_mode((dimx*cellsize, dimy*cellsize))

    if option == 1:
        pygame.display.set_caption("Równanie falowe")

    elif option == 2:
        pygame.display.set_caption("Równanie falowe - 2 punkty")

    elif option == 3:
        pygame.display.set_caption("Równanie falowe - z przeszkodą")

    elif option == 4:
        pygame.display.set_caption("Równanie falowe - z 2 przeszkodami")

    u, alpha = init_simulation(option)
    pixeldata = np.zeros((dimx, dimy, 3), dtype=np.uint8)

    while True:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                place_raindrops(u, pos)

        tick = tick + 1
        current_time = time.time()

        if current_time - start_time > 0.5:

            start_time = time.time()

            if option == 1:
                place_raindrops(u, [400, 400])

            elif option == 2:
                place_raindrops(u, [200, 400])
                place_raindrops(u, [600, 400])

            elif option == 3:
                place_raindrops(u, [400, 200])

            elif option == 4:
                place_raindrops(u, [400, 200])

        update(u, alpha)

        pixeldata[1:dimx, 1:dimy, 0] = np.clip(u[0, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 1] = np.clip(u[1, 1:dimx, 1:dimy] + 128, 0, 255)
        pixeldata[1:dimx, 1:dimy, 2] = np.clip(u[2, 1:dimx, 1:dimy] + 128, 0, 255)

        surf = pygame.surfarray.make_surface(pixeldata)
        display.blit(pygame.transform.scale(surf, (dimx * cellsize, dimy * cellsize)), (0, 0))

        if option == 3 or option == 4:
            pygame.draw.line(display, (0, 0, 0), (0, 400), (196 * 2, 400), 1)
            pygame.draw.line(display, (0, 0, 0), (204 * 2, 400), (400 * 2, 400), 1)

        if option == 4:
            pygame.draw.line(display, (0, 0, 0), (0, 600), (128 * 2, 600), 1)
            pygame.draw.line(display, (0, 0, 0), (136 * 2, 600), (264 * 2, 600), 1)
            pygame.draw.line(display, (0, 0, 0), (272 * 2, 600), (400 * 2, 600), 1)

        pygame.display.update()


if __name__ == "__main__":
    main(4)

# 1 - 1 punkt
# 2 - 2 punkty
# 3 - z przeszkodą
# 4 - z 2 przeszkodami
