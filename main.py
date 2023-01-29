import pygame
import numpy as np
import math
import time
import sympy

hs = 1
ts = 1
dimx = 400
dimy = 400
cellsize = 2


def create_arrays():
    global velocity
    global tau
    global kappa
    global gauss_peak
    global u

    u = np.zeros((3, dimx, dimy))

    velocity = np.zeros((dimx, dimy))

    tau = np.zeros((dimx, dimy))

    kappa = np.zeros((dimx, dimy))

    sz = 10
    sigma = 3
    xx, yy = np.meshgrid(range(-sz, sz), range(-sz, sz))

    gauss_peak = np.zeros((sz, sz))
    gauss_peak = 400 / (sigma * 2 * math.pi) * (math.sqrt(2 * math.pi)) * np.exp(
        - 0.5 * ((xx ** 2 + yy ** 2) / (sigma ** 2)))


def set_initial_conditions(u, pos, option):
    global tau
    global velocity
    global kappa
    global gauss_peak

    velocity[0:dimx, 0:dimy] = 0.5

    if option == 1:
        velocity[0:196, dimy - 200] = 0
        velocity[204:400, dimy - 200] = 0

    if option == 2:
        velocity[0:196, dimy - 200] = 0
        velocity[204:400, dimy - 200] = 0

        velocity[0:128, dimy - 100] = 0
        velocity[136:264, dimy - 100] = 0
        velocity[272:400, dimy - 100] = 0

    tau = ((velocity * ts) / hs) ** 2
    kappa = ts * velocity / hs

    put_gauss_peak(u, int(pos[0] / 2), int(pos[1] / 2), 10)


def put_gauss_peak(u, x: int, y: int, height):
    w, h = gauss_peak.shape
    w = int(w / 2)
    h = int(h / 2)

    use_multipole = False
    if use_multipole:

        dist = 3
        u[0:2, x - w - dist:x + w - dist, y - h:y + h] += height * gauss_peak
        u[0:2, x - w:x + w, y - h + dist:y + h + dist] -= height * gauss_peak
        u[0:2, x - w + dist:x + w + dist, y - h:y + h] += height * gauss_peak
        u[0:2, x - w:x + w, y - h - dist:y + h - dist] -= height * gauss_peak
    else:
        u[0:2, x - w:x + w, y - h:y + h] += height * gauss_peak


def update(u: any, method: int):
    u[2] = u[1]
    u[1] = u[0]

    boundary_size = 1
    if method == 0:

        u[0, 1:dimx - 1, 1:dimy - 1] = tau[1:dimx - 1, 1:dimy - 1] \
                                       * (0.25 * u[1, 0:dimx - 2, 0:dimy - 2]
                                          + 0.5 * u[1, 1:dimx - 1, 0:dimy - 2]
                                          + 0.25 * u[1, 2:dimx, 0:dimy - 2]

                                          + 0.5 * u[1, 0:dimx - 2, 1:dimy - 1]
                                          - 3 * u[1, 1:dimx - 1, 1:dimy - 1]
                                          + 0.5 * u[1, 2:dimx, 1:dimy - 1]

                                          + 0.25 * u[1, 0:dimx - 2, 2:dimy]
                                          + 0.5 * u[1, 1:dimx - 1, 2:dimy]
                                          + 0.25 * u[1, 2:dimx, 2:dimy]
                                          ) \
                                       + 2 * u[1, 1:dimx - 1, 1:dimy - 1] \
                                       - u[2, 1:dimx - 1, 1:dimy - 1]
    mur = True
    if mur:
        update_boundary(u, boundary_size)


def update_boundary(u, sz) -> None:
    c = dimx - 1
    u[0, dimx - sz - 1:c, 1:dimy - 1] = u[1, dimx - sz - 2:c - 1, 1:dimy - 1] + (
            kappa[dimx - sz - 1:c, 1:dimy - 1] - 1) / (kappa[dimx - sz - 1:c, 1:dimy - 1] + 1) * (
                                                u[0, dimx - sz - 2:c - 1, 1:dimy - 1] - u[1, dimx - sz - 1:c,
                                                                                        1:dimy - 1])

    c = 0
    u[0, c:sz, 1:dimy - 1] = u[1, c + 1:sz + 1, 1:dimy - 1] + (kappa[c:sz, 1:dimy - 1] - 1) / (
            kappa[c:sz, 1:dimy - 1] + 1) * (u[0, c + 1:sz + 1, 1:dimy - 1] - u[1, c:sz, 1:dimy - 1])

    r = dimy - 1
    u[0, 1:dimx - 1, dimy - 1 - sz:r] = u[1, 1:dimx - 1, dimy - 2 - sz:r - 1] + (
            kappa[1:dimx - 1, dimy - 1 - sz:r] - 1) / (kappa[1:dimx - 1, dimy - 1 - sz:r] + 1) * (
                                                u[0, 1:dimx - 1, dimy - 2 - sz:r - 1] - u[1, 1:dimx - 1,
                                                                                        dimy - 1 - sz:r])

    r = 0
    u[0, 1:dimx - 1, r:sz] = u[1, 1:dimx - 1, r + 1:sz + 1] + (kappa[1:dimx - 1, r:sz] - 1) / (
            kappa[1:dimx - 1, r:sz] + 1) * (u[0, 1:dimx - 1, r + 1:sz + 1] - u[1, 1:dimx - 1, r:sz])


def place_raindrops(u, pos, h):
    put_gauss_peak(u, int(pos[0] / 2), int(pos[1] / 2), h)


def draw_waves(display, u, data, offset):
    global velocity
    global font

    data[1:dimx, 1:dimy, 0] = 255 - np.clip(
        (u[0, 1:dimx, 1:dimy] > 0) * 10 * u[0, 1:dimx, 1:dimy] + u[1, 1:dimx, 1:dimy] + u[2, 1:dimx, 1:dimy], 0, 255)
    data[1:dimx, 1:dimy, 1] = 255 - np.clip(np.abs(u[0, 1:dimx, 1:dimy]) * 10, 0, 255)
    data[1:dimx, 1:dimy, 2] = 255 - np.clip(
        (u[0, 1:dimx, 1:dimy] <= 0) * -10 * u[0, 1:dimx, 1:dimy] + u[1, 1:dimx, 1:dimy] + u[2, 1:dimx, 1:dimy], 0, 255)

    surf = pygame.surfarray.make_surface(data)
    display.blit(pygame.transform.scale(surf, (dimx * cellsize, dimy * cellsize)), offset)


def main(option):
    global font

    pygame.init()
    pygame.font.init()

    font = pygame.font.SysFont('Consolas', 15)
    display = pygame.display.set_mode((dimx * cellsize, dimy * cellsize))

    create_arrays()

    if option == 1:
        pygame.display.set_caption('Równanie falowe')
        set_initial_conditions(u, [400, 400], 0)

    elif option == 2:
        pygame.display.set_caption('Równanie falowe - 2 punkty obok siebie')
        set_initial_conditions(u, [200, 400], 0)
        set_initial_conditions(u, [600, 400], 0)

    elif option == 3:
        pygame.display.set_caption('Równanie falowe - 1 punkt z przeszkodą')
        set_initial_conditions(u, [400, 200], 1)

    elif option == 4:
        pygame.display.set_caption('Równanie falowe - 1 punkt z 2 przeszkodami')
        set_initial_conditions(u, [400, 200], 2)

    image1data = np.zeros((dimx, dimy, 3), dtype=np.uint8)

    tick = 0
    last_tick = 0
    start_time = time.time()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.MOUSEBUTTONUP:
                pos = pygame.mouse.get_pos()
                place_raindrops(u, pos, 4)

        tick = tick + 1

        current_time = time.time()
        if current_time - start_time > 0.5:
            fps = (tick - last_tick) / (current_time - start_time)
            start_time = time.time()
            last_tick = tick
            if option == 1:
                place_raindrops(u, [400, 400], 10)
            elif option == 2:
                place_raindrops(u, [200, 400], 10)
                place_raindrops(u, [600, 400], 10)
            elif option == 3:
                place_raindrops(u, [400, 200], 10)
            elif option == 4:
                place_raindrops(u, [400, 200], 10)

        update(u, 0)
        draw_waves(display, u, image1data, (0, 0))

        pygame.display.update()


if __name__ == "__main__":
    main(2)

# 1 - 1 punkt
# 2 - 2 punkty obok siebie
# 3 - punkt z 1 przeszkodą
# 4 - punkt z 2 przeszkodami
