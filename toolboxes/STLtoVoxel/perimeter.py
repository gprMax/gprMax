from functools import reduce


def lines_to_voxels(line_list, pixels):
    current_line_indices = set()
    x = 0
    for event_x, status, line_ind in generate_line_events(line_list):
        while event_x - x >= 0:
            lines = reduce(lambda acc, cur: acc + [line_list[cur]], current_line_indices, [])
            paint_y_axis(lines, pixels, x)
            x += 1

        if status == "start":
            assert line_ind not in current_line_indices
            current_line_indices.add(line_ind)
        elif status == "end":
            assert line_ind in current_line_indices
            current_line_indices.remove(line_ind)


def slope_intercept(p1, p2):
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1

    return slope, intercept


def generate_y(p1, p2, x):
    slope, intercept = slope_intercept(p1, p2)
    y = slope * x + intercept

    return y


def paint_y_axis(lines, pixels, x):
    is_black = False
    target_ys = list(map(lambda line: int(generate_y(line[0], line[1], x)), lines))
    target_ys.sort()
    if len(target_ys) % 2:
        distances = []
        for i in range(len(target_ys) - 1):
            distances.append(target_ys[i + 1] - target_ys[i])
        # https://stackoverflow.com/a/17952763
        min_idx = -min((x, -i) for i, x in enumerate(distances))[1]
        del target_ys[min_idx]

    yi = 0
    for target_y in target_ys:
        if is_black:
            # Bulk assign all pixels between yi -> target_y
            pixels[yi:target_y, x] = True
        pixels[target_y][x] = True
        is_black = not is_black
        yi = target_y
    assert is_black is False, "an error has occured at x%s" % x


def generate_line_events(line_list):
    events = []

    for i, line in enumerate(line_list):
        first, second = sorted(line, key=lambda pt: pt[0])
        events.append((first[0], "start", i))
        events.append((second[0], "end", i))

    return sorted(events, key=lambda tup: tup[0])
