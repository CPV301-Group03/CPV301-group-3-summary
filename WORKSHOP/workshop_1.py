import pygame
import sys
import math


def workshop1():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Geometric Primitives and Transformations")
    clock = pygame.time.Clock()

    WHITE = (255, 255, 255)
    RED = (255, 0, 0)

    start_point = None
    end_point = None
    current_rect = []

    def draw_rectangle(points, color=RED):
        if len(points) == 4:
            pygame.draw.polygon(screen, color, points, 2)

    def get_corners(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def translate(points, dx, dy):
        return [(x + dx, y + dy) for x, y in points]

    def rotate(points, angle_deg):
        angle_rad = math.radians(angle_deg)
        cx = sum([p[0] for p in points]) / 4
        cy = sum([p[1] for p in points]) / 4
        rotated = []
        for x, y in points:
            x -= cx
            y -= cy
            x_new = x * math.cos(angle_rad) - y * math.sin(angle_rad)
            y_new = x * math.sin(angle_rad) + y * math.cos(angle_rad)
            rotated.append((x_new + cx, y_new + cy))
        return rotated

    def scale(points, scale_x, scale_y):
        cx = sum([p[0] for p in points]) / 4
        cy = sum([p[1] for p in points]) / 4
        scaled = []
        for x, y in points:
            new_x = cx + (x - cx) * scale_x
            new_y = cy + (y - cy) * scale_y
            scaled.append((new_x, new_y))
        return scaled

    # Main Loop
    running = True
    screen.fill(WHITE)
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                start_point = pygame.mouse.get_pos()

            elif event.type == pygame.MOUSEBUTTONUP:
                end_point = pygame.mouse.get_pos()
                current_rect = get_corners(start_point, end_point)

            elif event.type == pygame.KEYDOWN and current_rect:
                if event.key == pygame.K_t:
                    dx = int(input("Enter dx for translation: "))
                    dy = int(input("Enter dy for translation: "))
                    current_rect = translate(current_rect, dx, dy)

                elif event.key == pygame.K_r:
                    angle = float(input("Enter angle for rotation (degrees): "))
                    current_rect = rotate(current_rect, angle)

                elif event.key == pygame.K_s:
                    scale_x = float(input("Enter x scale factor: "))
                    scale_y = float(input("Enter y scale factor: "))
                    current_rect = scale(current_rect, scale_x, scale_y)

        screen.fill(WHITE)
        if current_rect:
            draw_rectangle(current_rect)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    workshop1()
