import matplotlib.colors as mcolors


def generate_pastel_colors(n_colors):
    # Base colors in HSV
    hues = [x/n_colors for x in range(n_colors)]
    # Convert to RGB after adding saturation and value for pastel effect
    colors = [mcolors.hsv_to_rgb([hue, 0.6, 0.9]) for hue in hues]
    # Convert RGB to hex
    hex_colors = [mcolors.to_hex(color) for color in colors]
    return hex_colors
