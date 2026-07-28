# Autonomous Cleaning Robot Simulator

An interactive 2D simulation of a domestic cleaning robot that combines systematic room coverage with cost-aware path navigation. Built with Python and Pygame.

## Why this project stands out

- **Hybrid path planning:** uses a boustrophedon (snake-pattern) coverage planner to clean rooms systematically and a weighted A* search to navigate to each target.
- **Energy-aware movement:** penalises turns and revisiting cleaned tiles, making routes more realistic than a shortest-path-only approach.
- **Simulated sensing:** visualises a 7 × 7 LIDAR scan around the robot and highlights nearby obstacles.
- **Complete mission cycle:** tracks coverage, steps, energy, room-level progress, route efficiency, and autonomous return to the charging dock.
- **Interactive visualisation:** supports pause, single-step execution, speed controls, planned-path overlays, cleaning trail, and live performance dashboard.

## Algorithms

| Component | Technique | Purpose |
| --- | --- | --- |
| Area coverage | Boustrophedon sweep / BFS fallback | Cleans each room methodically before progressing |
| Navigation | Weighted A* | Selects paths while discouraging revisits and unnecessary room exits |
| Accessibility | Flood fill | Identifies the reachable cleaning area at startup |
| Obstacle awareness | Simulated 7 × 7 LIDAR | Detects and visualises nearby walls, furniture, and table legs |

## Run locally

**Prerequisite:** Python 3.10 or newer.

```bash
git clone https://github.com/YOUR-USERNAME/autonomous-cleaning-robot.git
cd autonomous-cleaning-robot
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python ROBOT_CLEANING.py
```

## Controls

| Key | Action |
| --- | --- |
| `Space` | Start or pause; restart after completion |
| `R` | Reset the simulation |
| `S` | Advance one simulation step |
| `+` / `-` | Increase or decrease movement speed |
| `>` / `<` | Increase or decrease steps per frame |
| `Q` or window close | Exit |

## Resume-ready description

> Developed an interactive autonomous cleaning-robot simulator in Python/Pygame, combining systematic boustrophedon coverage planning with weighted A* navigation. Modelled LIDAR obstacle sensing, turn/revisit energy costs, room-level progress, and autonomous charging-dock return through a live visual dashboard.

**Skills:** Python, Pygame, A* search, BFS, graph traversal, heuristic search, simulation, UI visualisation.

## Project structure

```text
.
├── ROBOT_CLEANING.py   # Simulation, algorithms, visualisation, and controls
├── requirements.txt    # Runtime dependency
├── LICENSE             # MIT License
└── README.md           # Setup and project overview
```

## Future enhancements

- Add selectable floor plans and randomly placed obstacles.
- Compare A* against Dijkstra's algorithm using path length and energy metrics.
- Export an end-of-run performance report to CSV.

## License

Released under the [MIT License](LICENSE).
