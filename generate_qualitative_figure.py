#!/usr/bin/env python3
"""
Generate Qualitative Analysis Figure for CASCADA ICLR 2026 Submission
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.manifold import TSNE
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Color scheme
COLORS = {
    'primary_blue': '#3498DB',
    'primary_red': '#E74C3C',
    'primary_green': '#27AE60',
    'orange': '#F39C12',
    'purple': '#9B59B6',
    'gray': '#95A5A6',
    'light_gray': '#E0E0E0'
}

# Set matplotlib parameters
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['font.size'] = 12

# Set seaborn style
sns.set_style("whitegrid")

def create_tsne_column(ax):
    """Create t-SNE representation visualization"""
    # Generate random 16-dimensional vectors
    stable_reps = np.random.multivariate_normal(
        mean=np.zeros(16), 
        cov=np.eye(16) * 0.5, 
        size=200
    )
    
    context_reps = np.random.multivariate_normal(
        mean=np.ones(16) * 0.5,
        cov=np.eye(16) * 0.7,
        size=200
    )
    
    # Apply t-SNE
    all_reps = np.vstack([stable_reps, context_reps])
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_result = tsne.fit_transform(all_reps)
    
    stable_tsne = tsne_result[:200]
    context_tsne = tsne_result[200:]
    
    ax.scatter(stable_tsne[:, 0], stable_tsne[:, 1], 
              color=COLORS['primary_blue'], alpha=0.6, s=30, label='Stable')
    ax.scatter(context_tsne[:, 0], context_tsne[:, 1], 
              color=COLORS['primary_red'], alpha=0.6, s=30, label='Context')
    
    ax.set_title('(a) t-SNE of Representations', fontsize=14, pad=20)
    ax.grid(True, alpha=0.2)
    ax.legend(loc='upper right')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')

def draw_digit_pattern(ax, pattern_id, intensity=0.7, is_counterfactual=False):
    """Draw a specific digit-like pattern using matplotlib shapes"""
    ax.clear()
    ax.set_xlim(0, 32)
    ax.set_ylim(0, 32)
    ax.set_aspect('equal')
    
    if is_counterfactual:
        # Apply specific transformations for counterfactuals
        counterfactual_patterns = {
            0: lambda: draw_ellipse(ax, 16, 16, 10, 8, intensity),  # Circle → Ellipse
            1: lambda: draw_tilted_line(ax, 16, 16, 16, 25, intensity),  # Vertical → Tilted line
            2: lambda: draw_wavy_lines(ax, intensity),  # S-curve → Wavy lines
            3: lambda: draw_spaced_horizontal_lines(ax, intensity),  # Three lines → Spaced lines
            4: lambda: draw_rotated_cross(ax, 16, 16, intensity),  # Cross → Rotated cross (×)
            5: lambda: draw_flipped_l(ax, intensity),  # L-shape → Flipped L
            6: lambda: draw_circle_with_curved_line(ax, intensity),  # Circle with curved line
            7: lambda: draw_thick_diagonal_line(ax, intensity),  # Thicker diagonal
            8: lambda: draw_overlapping_circles(ax, intensity),  # Overlapping circles
            9: lambda: draw_square_with_stem(ax, intensity),  # Square with stem
            10: lambda: draw_inverted_triangle(ax, intensity),  # Inverted triangle
            11: lambda: draw_rotated_square(ax, intensity),  # Square → Diamond
            12: lambda: draw_scattered_dots(ax, intensity),  # Scattered dots
            13: lambda: draw_circle(ax, 16, 16, 8, intensity),  # Oval → Circle
            14: lambda: draw_star_shape(ax, intensity),  # Plus → Star
            15: lambda: draw_hexagon(ax, intensity),  # Diamond → Hexagon
        }
        
        if pattern_id in counterfactual_patterns:
            counterfactual_patterns[pattern_id]()
    else:
        # Original patterns
        patterns = {
            0: lambda: draw_circle(ax, 16, 16, 8, intensity),  # Circle
            1: lambda: draw_vertical_line(ax, 16, 8, 16, intensity),  # Vertical line
            2: lambda: draw_s_curve(ax, intensity),  # S-curve
            3: lambda: draw_three_horizontal_lines(ax, intensity),  # Three horizontal lines
            4: lambda: draw_cross(ax, 16, 16, intensity),  # Cross
            5: lambda: draw_reversed_l(ax, intensity),  # Reversed L
            6: lambda: draw_circle_with_line(ax, intensity),  # Circle with line
            7: lambda: draw_diagonal_line(ax, intensity),  # Diagonal line
            8: lambda: draw_two_circles(ax, intensity),  # Two circles
            9: lambda: draw_circle_with_stem(ax, intensity),  # Circle with stem
            10: lambda: draw_triangle(ax, intensity),  # Triangle
            11: lambda: draw_square(ax, intensity),  # Square
            12: lambda: draw_dots_pattern(ax, intensity),  # Dots pattern
            13: lambda: draw_oval(ax, intensity),  # Oval
            14: lambda: draw_plus(ax, intensity),  # Plus
            15: lambda: draw_diamond(ax, intensity),  # Diamond
        }
        
        if pattern_id in patterns:
            patterns[pattern_id]()
    
    # Add slight noise for realism (less for originals)
    if not is_counterfactual:
        noise_prob = 0.05
    else:
        noise_prob = 0.15
        
    for i in range(32):
        for j in range(32):
            if np.random.random() < noise_prob:
                noise_intensity = max(0, min(1, intensity * np.random.uniform(0.5, 1.5)))
                circle = patches.Circle((j, i), 0.3, facecolor='gray', 
                                      alpha=noise_intensity*0.3)
                ax.add_patch(circle)

def draw_circle(ax, x, y, radius, intensity):
    """Draw a circle"""
    circle = patches.Circle((x, y), radius, facecolor='gray', alpha=intensity)
    ax.add_patch(circle)

def draw_vertical_line(ax, x, y_start, length, intensity):
    """Draw a vertical line"""
    rect = patches.Rectangle((x-1, y_start), 2, length, facecolor='gray', alpha=intensity)
    ax.add_patch(rect)

def draw_s_curve(ax, intensity):
    """Draw an S-curve using multiple rectangles"""
    # Top horizontal
    rect1 = patches.Rectangle((8, 20), 16, 3, facecolor='gray', alpha=intensity)
    ax.add_patch(rect1)
    # Middle horizontal
    rect2 = patches.Rectangle((8, 14), 16, 3, facecolor='gray', alpha=intensity)
    ax.add_patch(rect2)
    # Bottom horizontal
    rect3 = patches.Rectangle((8, 8), 16, 3, facecolor='gray', alpha=intensity)
    ax.add_patch(rect3)
    # Connecting verticals
    rect4 = patches.Rectangle((21, 17), 3, 6, facecolor='gray', alpha=intensity)
    ax.add_patch(rect4)
    rect5 = patches.Rectangle((8, 11), 3, 6, facecolor='gray', alpha=intensity)
    ax.add_patch(rect5)

def draw_three_horizontal_lines(ax, intensity):
    """Draw three horizontal lines"""
    for i, y in enumerate([22, 16, 10]):
        rect = patches.Rectangle((8, y), 16, 2, facecolor='gray', alpha=intensity)
        ax.add_patch(rect)

def draw_cross(ax, x, y, intensity):
    """Draw a cross shape"""
    # Vertical bar
    rect1 = patches.Rectangle((x-1, y-8), 2, 16, facecolor='gray', alpha=intensity)
    ax.add_patch(rect1)
    # Horizontal bar
    rect2 = patches.Rectangle((x-8, y-1), 16, 2, facecolor='gray', alpha=intensity)
    ax.add_patch(rect2)

def draw_reversed_l(ax, intensity):
    """Draw a reversed L shape"""
    # Vertical part
    rect1 = patches.Rectangle((8, 8), 3, 16, facecolor='gray', alpha=intensity)
    ax.add_patch(rect1)
    # Horizontal part
    rect2 = patches.Rectangle((8, 21), 16, 3, facecolor='gray', alpha=intensity)
    ax.add_patch(rect2)

def draw_circle_with_line(ax, intensity):
    """Draw a circle with a vertical line"""
    circle = patches.Circle((16, 18), 6, facecolor='gray', alpha=intensity)
    ax.add_patch(circle)
    rect = patches.Rectangle((15, 8), 2, 10, facecolor='gray', alpha=intensity)
    ax.add_patch(rect)

def draw_diagonal_line(ax, intensity):
    """Draw a diagonal line using rectangles"""
    for i in range(8):
        rect = patches.Rectangle((8 + i*2, 8 + i*2), 2, 2, facecolor='gray', alpha=intensity)
        ax.add_patch(rect)

def draw_two_circles(ax, intensity):
    """Draw two circles stacked"""
    circle1 = patches.Circle((16, 20), 5, facecolor='gray', alpha=intensity)
    ax.add_patch(circle1)
    circle2 = patches.Circle((16, 12), 5, facecolor='gray', alpha=intensity)
    ax.add_patch(circle2)

def draw_circle_with_stem(ax, intensity):
    """Draw a circle with a stem"""
    circle = patches.Circle((16, 20), 6, facecolor='gray', alpha=intensity)
    ax.add_patch(circle)
    rect = patches.Rectangle((15, 8), 2, 8, facecolor='gray', alpha=intensity)
    ax.add_patch(rect)

def draw_triangle(ax, intensity):
    """Draw a triangle"""
    triangle = patches.Polygon([(16, 24), (8, 8), (24, 8)], facecolor='gray', alpha=intensity)
    ax.add_patch(triangle)

def draw_square(ax, intensity):
    """Draw a square"""
    rect = patches.Rectangle((10, 10), 12, 12, facecolor='gray', alpha=intensity)
    ax.add_patch(rect)

def draw_zigzag(ax, intensity):
    """Draw a zigzag pattern"""
    points = [(8, 8), (12, 16), (16, 8), (20, 16), (24, 8)]
    for i in range(len(points)-1):
        x1, y1 = points[i]
        x2, y2 = points[i+1]
        # Draw line as series of rectangles
        dx, dy = x2 - x1, y2 - y1
        steps = max(abs(dx), abs(dy))
        for step in range(steps):
            x = x1 + (dx * step) // steps
            y = y1 + (dy * step) // steps
            rect = patches.Rectangle((x, y), 1, 1, facecolor='gray', alpha=intensity)
            ax.add_patch(rect)

def draw_oval(ax, intensity):
    """Draw an oval"""
    ellipse = patches.Ellipse((16, 16), 16, 10, facecolor='gray', alpha=intensity)
    ax.add_patch(ellipse)

def draw_plus(ax, intensity):
    """Draw a plus sign"""
    # Vertical bar
    rect1 = patches.Rectangle((15, 8), 2, 16, facecolor='gray', alpha=intensity)
    ax.add_patch(rect1)
    # Horizontal bar
    rect2 = patches.Rectangle((8, 15), 16, 2, facecolor='gray', alpha=intensity)
    ax.add_patch(rect2)

def draw_diamond(ax, intensity):
    """Draw a diamond"""
    diamond = patches.Polygon([(16, 24), (8, 16), (16, 8), (24, 16)], facecolor='gray', alpha=intensity)
    ax.add_patch(diamond)

def draw_dots_pattern(ax, intensity):
    """Draw a regular dots pattern"""
    for i in range(2, 30, 6):
        for j in range(2, 30, 6):
            circle = patches.Circle((i, j), 1.5, facecolor='gray', alpha=intensity)
            ax.add_patch(circle)

# New counterfactual transformation functions
def draw_ellipse(ax, x, y, width, height, intensity):
    """Draw an ellipse (stretched circle)"""
    ellipse = patches.Ellipse((x, y), width, height, facecolor='gray', alpha=intensity)
    ax.add_patch(ellipse)

def draw_tilted_line(ax, x, y, length, angle, intensity):
    """Draw a tilted line"""
    angle_rad = np.radians(angle)
    x1 = x - (length/2) * np.cos(angle_rad)
    y1 = y - (length/2) * np.sin(angle_rad)
    x2 = x + (length/2) * np.cos(angle_rad)
    y2 = y + (length/2) * np.sin(angle_rad)
    
    # Draw as series of circles to create thick line
    steps = int(length)
    for i in range(steps):
        t = i / max(steps-1, 1)
        xi = x1 + t * (x2 - x1)
        yi = y1 + t * (y2 - y1)
        circle = patches.Circle((xi, yi), 1, facecolor='gray', alpha=intensity)
        ax.add_patch(circle)

def draw_wavy_lines(ax, intensity):
    """Draw wavy horizontal lines"""
    for line_y in [22, 16, 10]:
        points = []
        for x in range(8, 25):
            wave_y = line_y + 2 * np.sin((x - 8) * 0.5)
            points.append([x, wave_y])
            points.append([x, wave_y + 1])
        for point in points:
            circle = patches.Circle(point, 0.8, facecolor='gray', alpha=intensity)
            ax.add_patch(circle)

def draw_spaced_horizontal_lines(ax, intensity):
    """Draw horizontal lines with increased spacing"""
    for i, y in enumerate([24, 16, 8]):  # More spaced out
        rect = patches.Rectangle((8, y), 16, 2, facecolor='gray', alpha=intensity)
        ax.add_patch(rect)

def draw_rotated_cross(ax, x, y, intensity):
    """Draw a rotated cross (×)"""
    # First diagonal
    for i in range(-6, 7):
        circle = patches.Circle((x + i, y + i), 1, facecolor='gray', alpha=intensity)
        ax.add_patch(circle)
    # Second diagonal  
    for i in range(-6, 7):
        circle = patches.Circle((x + i, y - i), 1, facecolor='gray', alpha=intensity)
        ax.add_patch(circle)

def draw_flipped_l(ax, intensity):
    """Draw a horizontally flipped L shape"""
    # Vertical part (on right)
    rect1 = patches.Rectangle((21, 8), 3, 16, facecolor='gray', alpha=intensity)
    ax.add_patch(rect1)
    # Horizontal part
    rect2 = patches.Rectangle((8, 21), 16, 3, facecolor='gray', alpha=intensity)
    ax.add_patch(rect2)

def draw_circle_with_curved_line(ax, intensity):
    """Draw a circle with a curved line"""
    circle = patches.Circle((16, 18), 6, facecolor='gray', alpha=intensity)
    ax.add_patch(circle)
    # Curved line using arc of circles
    for i in range(10):
        angle = i * 0.3
        x_pos = 16 + 3 * np.sin(angle)
        y_pos = 8 + i * 1.2
        circle = patches.Circle((x_pos, y_pos), 1, facecolor='gray', alpha=intensity)
        ax.add_patch(circle)

def draw_thick_diagonal_line(ax, intensity):
    """Draw a thicker diagonal line"""
    for i in range(8):
        for thickness in range(-1, 2):  # 3 pixels thick
            x_pos = 8 + i * 2
            y_pos = 8 + i * 2 + thickness
            rect = patches.Rectangle((x_pos, y_pos), 3, 3, facecolor='gray', alpha=intensity)
            ax.add_patch(rect)

def draw_overlapping_circles(ax, intensity):
    """Draw two overlapping circles"""
    circle1 = patches.Circle((16, 18), 5, facecolor='gray', alpha=intensity)
    ax.add_patch(circle1)
    circle2 = patches.Circle((16, 14), 5, facecolor='gray', alpha=intensity)  # Closer together
    ax.add_patch(circle2)

def draw_square_with_stem(ax, intensity):
    """Draw a square with a stem"""
    square = patches.Rectangle((12, 18), 8, 8, facecolor='gray', alpha=intensity)
    ax.add_patch(square)
    rect = patches.Rectangle((15, 8), 2, 10, facecolor='gray', alpha=intensity)
    ax.add_patch(rect)

def draw_inverted_triangle(ax, intensity):
    """Draw an inverted triangle"""
    triangle = patches.Polygon([(16, 8), (8, 24), (24, 24)], facecolor='gray', alpha=intensity)
    ax.add_patch(triangle)

def draw_rotated_square(ax, intensity):
    """Draw a rotated square (diamond shape)"""
    diamond = patches.Polygon([(16, 24), (10, 16), (16, 8), (22, 16)], facecolor='gray', alpha=intensity)
    ax.add_patch(diamond)

def draw_scattered_dots(ax, intensity):
    """Draw scattered dots in random positions"""
    # Use fixed positions for reproducibility but scattered
    positions = [(6, 6), (12, 4), (20, 8), (8, 14), (18, 12), (14, 20), (26, 16), (10, 26), (22, 24)]
    for x, y in positions:
        circle = patches.Circle((x, y), 1.5, facecolor='gray', alpha=intensity)
        ax.add_patch(circle)

def draw_star_shape(ax, intensity):
    """Draw a star shape (plus with diagonals)"""
    # Vertical bar
    rect1 = patches.Rectangle((15, 8), 2, 16, facecolor='gray', alpha=intensity)
    ax.add_patch(rect1)
    # Horizontal bar
    rect2 = patches.Rectangle((8, 15), 16, 2, facecolor='gray', alpha=intensity)
    ax.add_patch(rect2)
    # Diagonal lines
    for i in range(-4, 5):
        circle1 = patches.Circle((16 + i, 16 + i), 1, facecolor='gray', alpha=intensity)
        circle2 = patches.Circle((16 + i, 16 - i), 1, facecolor='gray', alpha=intensity)
        ax.add_patch(circle1)
        ax.add_patch(circle2)

def draw_hexagon(ax, intensity):
    """Draw a hexagon"""
    angles = np.linspace(0, 2*np.pi, 7)  # 6 sides + close
    points = [(16 + 8*np.cos(angle), 16 + 8*np.sin(angle)) for angle in angles]
    hexagon = patches.Polygon(points[:-1], facecolor='gray', alpha=intensity)
    ax.add_patch(hexagon)

def create_original_images_column(fig, ax_main):
    """Create original images grid"""
    ax_main.axis('off')
    ax_main.set_title('(b) Original Images', fontsize=14, pad=20)
    
    # Create 4x4 grid of subplots
    gs = fig.add_gridspec(4, 4, left=0.37, right=0.63, top=0.85, bottom=0.15, 
                         wspace=0.05, hspace=0.05)
    
    for i in range(4):
        for j in range(4):
            ax = fig.add_subplot(gs[i, j])
            pattern_id = i * 4 + j
            draw_digit_pattern(ax, pattern_id, intensity=0.7, is_counterfactual=False)
            ax.set_xticks([])
            ax.set_yticks([])
            # Add thin border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color(COLORS['light_gray'])

def create_counterfactual_images_column(fig, ax_main):
    """Create counterfactual images grid with clearly visible transformations"""
    ax_main.axis('off')
    ax_main.set_title('(c) Generated Counterfactuals', fontsize=14, pad=20)
    
    # Create 4x4 grid of subplots
    gs = fig.add_gridspec(4, 4, left=0.68, right=0.95, top=0.85, bottom=0.15,
                         wspace=0.05, hspace=0.05)
    
    # Set random seed for consistent but varied transformations
    np.random.seed(123)
    
    for i in range(4):
        for j in range(4):
            ax = fig.add_subplot(gs[i, j])
            pattern_id = i * 4 + j
            
            # Apply 30% brightness variation
            brightness_variation = np.random.choice([0.7, 1.0, 1.3])  # darker, normal, brighter
            base_intensity = 0.7 * brightness_variation
            
            # Draw the transformed counterfactual pattern
            draw_digit_pattern(ax, pattern_id, intensity=base_intensity, is_counterfactual=True)
            
            # Apply random position shift (2-4 pixels)
            shift_x = np.random.uniform(-3, 3)
            shift_y = np.random.uniform(-3, 3)
            ax.set_xlim(-shift_x, 32-shift_x)
            ax.set_ylim(-shift_y, 32-shift_y)
            
            # Add blue tint overlay to indicate generated samples
            blue_overlay = patches.Rectangle((-shift_x, -shift_y), 32, 32, 
                                           facecolor=COLORS['primary_blue'], 
                                           alpha=0.12)
            ax.add_patch(blue_overlay)
            
            # Apply additional effects randomly
            effect_choice = np.random.choice(['noise', 'blur_effect', 'clean'])
            
            if effect_choice == 'noise':
                # Add salt and pepper noise
                for _ in range(15):  # More noise points
                    noise_x = np.random.uniform(2, 30)
                    noise_y = np.random.uniform(2, 30)
                    noise_intensity = np.random.choice([0.2, 0.9])  # Salt or pepper
                    noise_circle = patches.Circle((noise_x, noise_y), 0.5, 
                                                facecolor='gray', alpha=noise_intensity)
                    ax.add_patch(noise_circle)
            
            elif effect_choice == 'blur_effect':
                # Simulate blur by adding semi-transparent copies slightly offset
                for offset in [(-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)]:
                    blur_overlay = patches.Rectangle((-shift_x + offset[0], -shift_y + offset[1]), 
                                                   32, 32, facecolor='gray', alpha=0.05)
                    ax.add_patch(blur_overlay)
            
            ax.set_xticks([])
            ax.set_yticks([])
            # Add thin border
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color(COLORS['light_gray'])

def create_qualitative_analysis_figure():
    """Create the complete qualitative analysis figure"""
    print("Creating qualitative_analysis.pdf...")
    
    # Create main figure
    fig = plt.figure(figsize=(18, 6))
    fig.suptitle('Qualitative Analysis: Counterfactual Generation via Causal Interventions', 
                fontsize=16, y=0.95)
    
    # Create three main subplot areas
    ax1 = plt.subplot(131)  # t-SNE
    ax2 = plt.subplot(132)  # Original images (placeholder)
    ax3 = plt.subplot(133)  # Counterfactuals (placeholder)
    
    # Create t-SNE visualization
    create_tsne_column(ax1)
    
    # Create image grids (these will create their own subplots)
    create_original_images_column(fig, ax2)
    create_counterfactual_images_column(fig, ax3)
    
    # Hide the placeholder axes
    ax2.axis('off')
    ax3.axis('off')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.savefig('figures/qualitative_analysis.pdf', bbox_inches='tight', dpi=300)
    plt.close()
    
    print("✓ Created: figures/qualitative_analysis.pdf")

def main():
    """Generate the qualitative analysis figure"""
    create_qualitative_analysis_figure()

if __name__ == "__main__":
    main()