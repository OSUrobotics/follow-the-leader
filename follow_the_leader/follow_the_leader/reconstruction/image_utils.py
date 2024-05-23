def get_contiguous_distance(pts, matches, vec):
    current_start = None
    dist = 0
    for i, match in enumerate(matches):
        if current_start is None and match:
            current_start = i
        elif current_start is not None and not match:
            offsets = pts[current_start + 1 : i] - pts[current_start : i - 1]
            dist += (offsets * vec).sum()
            current_start = None
    if current_start is not None:
        i = len(matches)
        offsets = pts[current_start + 1 : i] - pts[current_start : i - 1]
        dist += (offsets * vec).sum()

    return dist