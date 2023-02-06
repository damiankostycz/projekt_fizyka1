import time
import pygame
import numpy as np

h = 1
k = 1
dimx = 400
dimy = 400
cellsize = 2

dx, dy = 0.01, 0.01
dt = 0.001
sigma_max = 20


npml = 400
sigma_x = np.zeros((dimx, dimy))
sigma_y = np.zeros((dimx, dimy))
for i in range(npml):
    x = i / npml
    sigma_x[i,:] = sigma_max * (x**4)
    sigma_x[dimx-1-i,:] = sigma_max * (x**4)
    sigma_y[:,i] = sigma_max * (x**4)
    sigma_y[:,dimy-1-i] = sigma_max * (x**4)


def init_simulation(option):
    global kappa
    u = np.zeros((3, dimx, dimy))
    c = 0.5
    alpha = np.zeros((dimx, dimy))
    alpha[0:dimx, 0:dimy] = ((c*k) / h)**2
    kappa = 1 * alpha / 1

    if option == 3 or option == 4:
        alpha[0:196, dimy - 200] = 0
        alpha[204:dimx, dimy - 200] = 0

    if option == 4:
        alpha[0:128, dimy - 100] = 0
        alpha[136:264, dimy - 100] = 0
        alpha[272:dimx, dimy - 100] = 0

    if option == 5:
        alpha[150:250, dimy - 200] = 0
        alpha[150:250, dimy - 100] = 0

        alpha[150, dimy - 200:dimy - 100] = 0
        alpha[250, dimy - 200:dimy - 100] = 0

    if option == 6:
        alpha[150:196, dimy - 200] = 0
        alpha[204:250, dimy - 200] = 0

        alpha[150:196, dimy - 100] = 0
        alpha[204:250, dimy - 100] = 0

        alpha[150, 200:246] = 0
        alpha[150, 254:300] = 0

        alpha[250, 200:246] = 0
        alpha[250, 254:300] = 0

    return u, alpha


def update(u, alpha):
    # u[2] = u[1]
    # u[1] = u[0]
    u[2, :, :] = u[1, :, :]
    u[1, :, :] = u[0, :, :]
    u[0, 1:dimx-1, 1:dimy-1] = alpha[1:dimx-1, 1:dimy-1] * (u[1, 0:dimx-2, 1:dimy-1] +
                                         u[1, 2:dimx,   1:dimy-1] +
                                         u[1, 1:dimx-1, 0:dimy-2] +
                                         u[1, 1:dimx-1, 2:dimy] - 4*u[1, 1:dimx-1, 1:dimy-1]) \
                                        + 2 * u[1, 1:dimx-1, 1:dimy-1] - u[2, 1:dimx-1, 1:dimy-1]


    u[0, 1:dimx-1, 1:dimy-1] *= 0.998

    boundary_size = 1
    u[0, :, :] += dt * (sigma_x[:, :] * (u[1, :, :] - u[0, :, :]) + sigma_y[:, :] * (u[1, :, :] - u[0, :, :]))

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

    elif option == 5:
        pygame.display.set_caption("Równanie falowe - z kwadratem")

    elif option == 6:
        pygame.display.set_caption("Równanie falowe - z kwadratem z dziurami")

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

            elif option == 3 or option == 5 or option == 6:
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

        if option == 5:
            pygame.draw.line(display, (0, 0, 0), (150 * 2, 400), (250 * 2, 400), 1)
            pygame.draw.line(display, (0, 0, 0), (150 * 2, 600), (250 * 2, 600), 1)

            pygame.draw.line(display, (0, 0, 0), (150 * 2, 400), (150 * 2, 600), 1)
            pygame.draw.line(display, (0, 0, 0), (250 * 2, 400), (250 * 2, 600), 1)

        if option == 6:
            pygame.draw.line(display, (0, 0, 0), (150 * 2, 400), (196 * 2, 400), 1)
            pygame.draw.line(display, (0, 0, 0), (150 * 2, 600), (196 * 2, 600), 1)

            pygame.draw.line(display, (0, 0, 0), (204 * 2, 400), (250 * 2, 400), 1)
            pygame.draw.line(display, (0, 0, 0), (204 * 2, 600), (250 * 2, 600), 1)

            pygame.draw.line(display, (0, 0, 0), (150 * 2, 400), (150 * 2, 246 * 2), 1)
            pygame.draw.line(display, (0, 0, 0), (150 * 2, 254 * 2), (150 * 2, 600), 1)

            pygame.draw.line(display, (0, 0, 0), (250 * 2, 400), (250 * 2, 246 * 2), 1)
            pygame.draw.line(display, (0, 0, 0), (250 * 2, 254 * 2), (250 * 2, 600), 1)

        pygame.display.update()


if __name__ == "__main__":
    main(6)

# 1 - 1 punkt
# 2 - 2 punkty
# 3 - z przeszkodą
# 4 - z 2 przeszkodami
# 5 - pusty kwadrat
# 6 - kwadrat z dziurami