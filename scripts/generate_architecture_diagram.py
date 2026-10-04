#!/usr/bin/env python3
"""
Generate a system architecture diagram for MOVE-OD Web Application
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(5, 9.5, "MOVE-OD Web Application Architecture", ha="center", va="top", fontsize=20, fontweight="bold")

    # User Browser
    browser = FancyBboxPatch(
        (3, 7.5), 4, 1, boxstyle="round,pad=0.1", edgecolor="#667eea", facecolor="#e8eaf6", linewidth=2
    )
    ax.add_patch(browser)
    ax.text(5, 8, "User Browser\nhttp://localhost:8080", ha="center", va="center", fontsize=11, fontweight="bold")

    # Frontend
    frontend = FancyBboxPatch(
        (0.5, 5), 3.5, 2, boxstyle="round,pad=0.1", edgecolor="#28a745", facecolor="#d4edda", linewidth=2
    )
    ax.add_patch(frontend)
    ax.text(2.25, 6.5, "Frontend", ha="center", fontsize=12, fontweight="bold")
    ax.text(2.25, 6.1, "Port: 8080", ha="center", fontsize=9)
    ax.text(2.25, 5.7, "• HTML/CSS", ha="center", fontsize=8)
    ax.text(2.25, 5.4, "• JavaScript", ha="center", fontsize=8)
    ax.text(2.25, 5.1, "• Leaflet.js", ha="center", fontsize=8)

    # Backend
    backend = FancyBboxPatch(
        (6, 5), 3.5, 2, boxstyle="round,pad=0.1", edgecolor="#dc3545", facecolor="#f8d7da", linewidth=2
    )
    ax.add_patch(backend)
    ax.text(7.75, 6.5, "Backend", ha="center", fontsize=12, fontweight="bold")
    ax.text(7.75, 6.1, "Port: 8000", ha="center", fontsize=9)
    ax.text(7.75, 5.7, "• FastAPI", ha="center", fontsize=8)
    ax.text(7.75, 5.4, "• Python", ha="center", fontsize=8)
    ax.text(7.75, 5.1, "• Processing", ha="center", fontsize=8)

    # Data Files
    data = FancyBboxPatch(
        (5.5, 2.5), 2, 1.8, boxstyle="round,pad=0.1", edgecolor="#ffc107", facecolor="#fff3cd", linewidth=2
    )
    ax.add_patch(data)
    ax.text(6.5, 3.8, "Data Files", ha="center", fontsize=11, fontweight="bold")
    ax.text(6.5, 3.4, "• LODES", ha="center", fontsize=8)
    ax.text(6.5, 3.1, "• Shapefiles", ha="center", fontsize=8)
    ax.text(6.5, 2.8, "• INRIX", ha="center", fontsize=8)

    # Job Queue
    queue = FancyBboxPatch(
        (8, 2.5), 1.8, 1.8, boxstyle="round,pad=0.1", edgecolor="#17a2b8", facecolor="#d1ecf1", linewidth=2
    )
    ax.add_patch(queue)
    ax.text(8.9, 3.8, "Job Queue", ha="center", fontsize=11, fontweight="bold")
    ax.text(8.9, 3.4, "• Status", ha="center", fontsize=8)
    ax.text(8.9, 3.1, "• Progress", ha="center", fontsize=8)
    ax.text(8.9, 2.8, "• Results", ha="center", fontsize=8)

    # Processing Modules
    modules = FancyBboxPatch(
        (0.5, 0.3), 6, 1.8, boxstyle="round,pad=0.1", edgecolor="#6c757d", facecolor="#e2e3e5", linewidth=2
    )
    ax.add_patch(modules)
    ax.text(3.5, 1.8, "Processing Modules (Existing)", ha="center", fontsize=11, fontweight="bold")
    ax.text(1.5, 1.4, "• LODES Read", ha="left", fontsize=8)
    ax.text(1.5, 1.1, "• Buildings", ha="left", fontsize=8)
    ax.text(1.5, 0.8, "• Locations", ha="left", fontsize=8)
    ax.text(1.5, 0.5, "• INRIX", ha="left", fontsize=8)

    ax.text(4, 1.4, "• Routing", ha="left", fontsize=8)
    ax.text(4, 1.1, "• Speed Shift", ha="left", fontsize=8)
    ax.text(4, 0.8, "• Calibration", ha="left", fontsize=8)
    ax.text(4, 0.5, "• ILP Solver", ha="left", fontsize=8)

    # Arrows
    # Browser to Frontend
    arrow1 = FancyArrowPatch((4.5, 7.5), (3, 7), arrowstyle="->", mutation_scale=20, linewidth=2, color="#667eea")
    ax.add_patch(arrow1)
    ax.text(3.5, 7.3, "HTTP", ha="center", fontsize=8, style="italic")

    # Browser to Backend
    arrow2 = FancyArrowPatch((5.5, 7.5), (7, 7), arrowstyle="->", mutation_scale=20, linewidth=2, color="#667eea")
    ax.add_patch(arrow2)
    ax.text(6.5, 7.3, "REST API", ha="center", fontsize=8, style="italic")

    # Frontend to Backend
    arrow3 = FancyArrowPatch((4, 6), (6, 6), arrowstyle="<->", mutation_scale=20, linewidth=2, color="#28a745")
    ax.add_patch(arrow3)
    ax.text(5, 6.3, "JSON", ha="center", fontsize=8, style="italic")

    # Backend to Data
    arrow4 = FancyArrowPatch((7.5, 5), (6.8, 4.3), arrowstyle="<->", mutation_scale=20, linewidth=2, color="#ffc107")
    ax.add_patch(arrow4)
    ax.text(7.3, 4.6, "Read/Write", ha="center", fontsize=7, style="italic")

    # Backend to Queue
    arrow5 = FancyArrowPatch((8, 5), (8.5, 4.3), arrowstyle="<->", mutation_scale=20, linewidth=2, color="#17a2b8")
    ax.add_patch(arrow5)
    ax.text(8.3, 4.6, "Track Jobs", ha="center", fontsize=7, style="italic")

    # Backend to Modules
    arrow6 = FancyArrowPatch((7, 5), (5, 2.1), arrowstyle="->", mutation_scale=20, linewidth=2, color="#dc3545")
    ax.add_patch(arrow6)
    ax.text(6.2, 3.5, "Process", ha="center", fontsize=8, style="italic")

    # Legend
    legend_y = 0.1
    ax.text(7.5, legend_y, "✅ New Components  |  📦 Existing Modules", ha="left", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig("move_od_architecture.png", dpi=300, bbox_inches="tight")
    print("✅ Architecture diagram saved as 'move_od_architecture.png'")

except ImportError:
    print("⚠️  matplotlib not installed. Skipping diagram generation.")
    print("Install with: pip install matplotlib")
