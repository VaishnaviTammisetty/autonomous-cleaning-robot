"""Interactive autonomous cleaning-robot path-planning simulator.

The robot covers each room using a systematic boustrophedon sweep, with BFS
fallback for missed cells.  Weighted A* navigates to the next target while
penalising turns and revisits; a simulated 7×7 LIDAR scan visualises nearby
obstacles.  Run this module directly to start the Pygame simulation.
"""

import pygame, sys, heapq, math, time
from collections import deque

# ═══════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════
CELL    = 22
COLS    = 48
ROWS    = 34
PANEL_W = 310
FPS     = 60

WIN_W = COLS * CELL + PANEL_W
WIN_H = ROWS * CELL

MOVE_COST       = 1.0
TURN_COST       = 0.4
REVISIT_PENALTY = 5.0
TRAIL_LIMIT     = 50

FREE      = 0
WALL      = 1
CLEANED   = 2
SOLID_OBS = 3
TABLE_LEG = 4

DIRS     = [(-1,0),(1,0),(0,-1),(0,1)]
PASSABLE = {FREE, CLEANED}

# ═══════════════════════════════════════════════════════════════
#  COLOUR PALETTE
# ═══════════════════════════════════════════════════════════════
BP = {
    "floor"       :(232,238,250), "floor_clean" :(175,212,245),
    "wall"        :( 22, 36, 72), "grid"        :(208,218,238),
    "furn_bg"     :(198,210,232), "furn_ln"     :( 68, 95,162),
    "furn_txt"    :( 22, 40, 88), "hatch"       :(165,180,215),
    "table_top"   :(205,218,240), "table_ln"    :( 68, 95,162),
    "leg"         :( 55, 85,155), "door_col"    :(160,195,240),
    "robot"       :( 28,188, 88), "robot_ring"  :(255,255,255),
    "robot_eye"   :(  8, 72, 28), "trail"       :( 92,155,215),
    "nav_path"    :(255,198, 48), "sensor"      :(255,160, 35),
    "panel_bg"    :( 15, 22, 48), "panel_div"   :( 55, 72,128),
    "txt_hi"      :(235,242,255), "txt_dim"     :(160,178,220),
    "bar_bg"      :( 38, 52, 98), "bar_cov"     :( 45,192,110),
    "bar_e_lo"    :( 55,125,215), "bar_e_mid"   :(235,145, 35),
    "bar_e_hi"    :(210, 45, 45), "bfs_col"     :( 75,195,255),
    "astar_col"   :(255,195, 55), "white"       :(255,255,255),
    "black"       :(  0,  0,  0), "room_lbl"    :( 55, 82,148),
    "warn"        :(255, 80, 80), "green_flash" :( 60,220,120),
    "charge_bg"   :(255,220,  0), "charge_ring" :(255,160,  0),
    "charge_bolt" :(255,255,255), "return_col"  :(255,120, 50),
    "grade_s"     :(255,215,  0), "grade_a"     :( 80,220,120),
    "grade_b"     :( 75,195,255), "grade_c"     :(235,185, 40),
    "grade_d"     :(220, 80, 80),
}

# ═══════════════════════════════════════════════════════════════
#  ROOM LAYOUT
# ═══════════════════════════════════════════════════════════════
ROOMS = [
    ("LIVING ROOM",     1,  1, 18, 24),
    ("MASTER BEDROOM",  1, 25, 18, 47),
    ("KITCHEN",        19,  1, 33, 17),
    ("DINING ROOM",    19, 18, 33, 32),
    ("BATHROOM",       19, 33, 33, 46),
]

# ═══════════════════════════════════════════════════════════════
#  FURNITURE
# ═══════════════════════════════════════════════════════════════
FURNITURE_DEFS = [
    # LIVING ROOM
    ("SOFA",         3,  2,  6, 10,  "solid"),
    ("COFFEE TBL",   8,  4, 10,  8,  "table"),
    ("TV UNIT",      6, 22, 14, 23,  "solid"),
    ("PLANT",        2,  2,  3,  3,  "solid"),
    ("PLANT",        2, 22,  3, 23,  "solid"),
    ("ARM CHAIR",   12,  2, 14,  5,  "solid"),
    ("ARM CHAIR",   12,  7, 14, 10,  "solid"),
    ("SIDE TABLE",  12, 12, 14, 14,  "table"),
    # MASTER BEDROOM
    ("DOUBLE BED",   3, 29,  9, 37,  "solid"),
    ("NIGHTSTAND",   3, 26,  5, 28,  "solid"),
    ("NIGHTSTAND",   3, 38,  5, 40,  "solid"),
    ("WARDROBE",     2, 41,  9, 47,  "solid"),
    ("STUDY DESK",  11, 26, 15, 32,  "table"),
    ("DESK CHAIR",  11, 33, 14, 35,  "passable"),
    ("BOOKSHELF",   12, 43, 17, 46,  "solid"),
    # KITCHEN
    ("COUNTER",     21,  2, 22, 11,  "solid"),
    ("STOVE",       23,  2, 25,  5,  "solid"),
    ("FRIDGE",      26,  2, 30,  5,  "solid"),
    ("SINK",        23,  8, 24, 11,  "solid"),
    ("KITCHEN ISL", 26,  7, 30, 13,  "table"),
    # DINING ROOM
    ("DINING TBL",  23, 22, 27, 28,  "passable"),
    ("CHAIR",       21, 24, 22, 26,  "passable"),
    ("CHAIR",       28, 24, 29, 26,  "passable"),
    ("CHAIR",       24, 20, 26, 21,  "passable"),
    ("CHAIR",       24, 29, 26, 30,  "passable"),
    # BATHROOM
    ("BATHTUB",     21, 34, 25, 40,  "solid"),
    ("TOILET",      26, 34, 29, 37,  "solid"),
    ("SINK",        26, 39, 27, 41,  "solid"),
    ("CABINET",     30, 34, 33, 43,  "solid"),
]

# ═══════════════════════════════════════════════════════════════
#  DOORWAYS
# ═══════════════════════════════════════════════════════════════
def _make_doorways():
    cells = []
    def hgap(r1,r2,cs,w):
        for r in range(r1,r2+1):
            for c in range(cs,cs+w): cells.append((r,c))
    def vgap(c1,c2,rs,h):
        for c in range(c1,c2+1):
            for r in range(rs,rs+h): cells.append((r,c))
    hgap(18,19, 7,4); hgap(18,19,20,4)
    hgap(18,19,27,4); hgap(18,19,39,4)
    vgap(17,18,24,4)
    return cells

DOORWAY_CELLS = _make_doorways()

# ═══════════════════════════════════════════════════════════════
#  PRECOMPUTED ROOM LOOKUP — O(1) room index per cell
# ═══════════════════════════════════════════════════════════════
_ROOM_GRID = [[-1]*COLS for _ in range(ROWS)]
for _i,(_, _r1,_c1,_r2,_c2) in enumerate(ROOMS):
    for _r in range(_r1,_r2+1):
        for _c in range(_c1,_c2+1):
            _ROOM_GRID[_r][_c] = _i

# ═══════════════════════════════════════════════════════════════
#  GRADE HELPER
# ═══════════════════════════════════════════════════════════════
def _grade(eff):
    if eff >= 9.0: return "S",  BP["grade_s"]
    if eff >= 7.0: return "A",  BP["grade_a"]
    if eff >= 5.0: return "B",  BP["grade_b"]
    if eff >= 3.0: return "C",  BP["grade_c"]
    return            "D",  BP["grade_d"]


# ═══════════════════════════════════════════════════════════════
#  SIMULATION CLASS
# ═══════════════════════════════════════════════════════════════
class Sim:

    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        pygame.display.set_caption("Cleaning Robot Path Finding Simulation")
        self.clock = pygame.time.Clock()
        self.ft  = pygame.font.SysFont("segoeui", 22, bold=True)
        self.fm  = pygame.font.SysFont("segoeui", 16, bold=True)
        self.fr  = pygame.font.SysFont("segoeui", 15, bold=True)
        self.fs  = pygame.font.SysFont("segoeui", 13, bold=True)
        self.fxs = pygame.font.SysFont("segoeui", 12, bold=True)
        self.move_delay  = 2
        self.steps_tick  = 1    # steps executed per timer tick (> key increases)
        self._bg_surface = None  # static pre-rendered background
        self.reset()

    # ═══════════════════════════════════════════════════════════
    #  WORLD BUILD
    # ═══════════════════════════════════════════════════════════
    def reset(self):
        self.grid        = [[FREE]*COLS for _ in range(ROWS)]
        self.table_cells = set()

        # Outer border
        for r in range(ROWS): self.grid[r][0] = self.grid[r][COLS-1] = WALL
        for c in range(COLS): self.grid[0][c] = self.grid[ROWS-1][c] = WALL
        for c in range(26,41): self.grid[2][c] = WALL

        # Room walls
        for _,r1,c1,r2,c2 in ROOMS:
            for r in range(r1,r2+1):
                for c in (c1,c2):
                    if 0<=r<ROWS and 0<=c<COLS: self.grid[r][c] = WALL
            for c in range(c1,c2+1):
                for r in (r1,r2):
                    if 0<=r<ROWS and 0<=c<COLS: self.grid[r][c] = WALL

        # Furniture
        for label,r1,c1,r2,c2,style in FURNITURE_DEFS:
            for r in range(r1,r2+1):
                for c in range(c1,c2+1):
                    if not (0<=r<ROWS and 0<=c<COLS): continue
                    if style=="solid": self.grid[r][c] = SOLID_OBS
                    elif style=="table":
                        self.table_cells.add((r,c))
                        if r in (r1,r2) and c in (c1,c2): self.grid[r][c] = TABLE_LEG
                    elif style=="passable":
                        # Robot can clean under these — cells stay FREE/CLEANED
                        # but are added to table_cells so furniture draws on top
                        self.table_cells.add((r,c))

        # Doorways last
        for r,c in DOORWAY_CELLS:
            if 0<=r<ROWS and 0<=c<COLS: self.grid[r][c] = FREE

        # Robot start = charging dock
        self.robot = None
        for r in range(1,ROWS-1):
            for c in range(1,COLS-1):
                if self.grid[r][c]==FREE: self.robot=[r,c]; break
            if self.robot: break
        self.start_pos = tuple(self.robot)
        self.grid[self.robot[0]][self.robot[1]] = CLEANED

        # Reachability + free-cell tracking
        self._reachable_set = self._flood_fill(tuple(self.robot))
        self.total_free     = len(self._reachable_set)
        # IMP-2: maintain set of dirty cells for O(1) "any left?" check
        self.free_cells     = {c for c in self._reachable_set
                               if self.grid[c[0]][c[1]] == FREE}

        # IMP-3: per-room cell counts for room progress tracking
        self._room_total   = [0]*len(ROOMS)   # total reachable cells per room
        self._room_cleaned = [0]*len(ROOMS)   # cleaned cells per room
        self._room_done    = [False]*len(ROOMS)
        for r,c in self._reachable_set:
            ri = _ROOM_GRID[r][c]
            if ri >= 0: self._room_total[ri] += 1
        # start cell already cleaned
        sr,sc = self.start_pos
        sri = _ROOM_GRID[sr][sc]
        if sri >= 0: self._room_cleaned[sri] = 1

        # IMP-8: room flash animation state {room_idx: frames_remaining}
        self._room_flash = {}

        self.nav_path        = []
        self.trail           = deque(maxlen=TRAIL_LIMIT)
        self.last_dir        = None
        self.energy          = 0.0
        self.cleaning_energy = 0.0
        self.return_energy   = 0.0
        self.steps           = 0
        self.running      = False
        self.finished     = False
        self.mtimer       = 0
        self.near_obs     = set()
        self.phase        = "BFS"
        self.bfs_ops      = 0
        self.astar_ops    = 0
        self.charge_anim  = 0
        self.anim_tick    = 0   # general animation counter (increments every frame)
        self.cleaned_count = 1

        # Pre-compute boustrophedon (snake) sweep order for each room.
        # Cells are visited row by row, alternating direction each row,
        # so the robot sweeps systematically with minimal backtracking.
        self._sweep_queues = self._build_sweep_plan()
        self._sweep_idx    = 0   # which room queue we are currently drawing from

        # IMP-4: elapsed time
        self._start_time  = None   # set on first step
        self._elapsed     = 0.0
        self._finish_time = None

        # Panel cache
        self._cov_cache          = self.cleaned_count/max(1,self.total_free)*100
        self._eff_cache          = 0.0   # cleaning efficiency (no return cost)
        self._overall_eff_cache  = 0.0   # full efficiency (includes return)
        self._dirty_panel        = True

        # IMP-1: invalidate static background
        self._bg_surface  = None

        self.msg = "SPACE=Start/Pause   R=Reset   +/-=Speed   S=Step   >=Fast"

    # ═══════════════════════════════════════════════════════════
    #  ALGORITHMS
    # ═══════════════════════════════════════════════════════════
    def _flood_fill(self, start):
        vis={start}; q=deque([start])
        while q:
            r,c=q.popleft()
            for dr,dc in DIRS:
                nb=(r+dr,c+dc)
                if nb in vis: continue
                nr,nc=nb
                if not (0<=nr<ROWS and 0<=nc<COLS): continue
                if self.grid[nr][nc] not in PASSABLE: continue
                vis.add(nb); q.append(nb)
        return vis

    def _build_sweep_plan(self):
        """
        Build boustrophedon (snake) sweep queues in the FIXED room order:
        Living Room → Kitchen → Dining Room → Master Bedroom → Bathroom.
        Each queue contains only cells inside that room.
        The robot will FULLY clean one room before starting the next.
        """
        FIXED_ORDER = [
            "LIVING ROOM",
            "KITCHEN",
            "DINING ROOM",
            "MASTER BEDROOM",
            "BATHROOM",
        ]
        # Build a lookup from name to room bounds
        room_map = {name: (r1,c1,r2,c2) for name,r1,c1,r2,c2 in ROOMS}
        self._sweep_room_names = FIXED_ORDER  # store for panel display

        queues = []
        for rname in FIXED_ORDER:
            r1,c1,r2,c2 = room_map[rname]
            q = deque(); row_num = 0
            for r in range(r1+1, r2):
                row_cells = [
                    (r, c) for c in range(c1+1, c2)
                    if self.grid[r][c] in PASSABLE
                ]
                if not row_cells: continue
                # Snake pattern: even rows L→R, odd rows R→L
                if row_num % 2 == 1:
                    row_cells = list(reversed(row_cells))
                q.extend(row_cells)
                row_num += 1
            queues.append(q)
        return queues

    def _sweep_target(self):
        """
        Pull the next uncleaned cell ONLY from the current room's queue.
        The robot is locked to finish one room completely before moving on.
        Falls back to in-room BFS if the queue has no FREE cells left,
        and only advances to the next room when this room is truly empty.
        """
        if not self.free_cells: return None

        while self._sweep_idx < len(self._sweep_queues):
            q = self._sweep_queues[self._sweep_idx]

            # Drain stale (already-cleaned) entries
            while q:
                r, c = q[0]
                if self.grid[r][c] == FREE:
                    q.popleft()
                    self.bfs_ops += 1
                    return (r, c)
                q.popleft()

            # Queue empty — check if any FREE cells remain in this room
            rname = self._sweep_room_names[self._sweep_idx]
            room_entry = next(e for e in ROOMS if e[0] == rname)
            _, r1, c1, r2, c2 = room_entry
            room_free = any(
                self.grid[r][c] == FREE
                for r in range(r1+1, r2)
                for c in range(c1+1, c2)
                if 0<=r<ROWS and 0<=c<COLS
            )
            if room_free:
                # BFS within this room only for any missed cell
                return self._bfs_in_room(r1, c1, r2, c2)

            # Room truly done — advance to next room
            self._sweep_idx += 1

        # All rooms done — catch any globally missed cells
        return self._bfs_fallback()

    def _bfs_in_room(self, r1, c1, r2, c2):
        """BFS restricted to cells inside the given room bounds."""
        start = tuple(self.robot)
        visited = {start}; queue = deque([start]); ops = 0
        while queue:
            r, c = queue.popleft(); ops += 1
            for dr, dc in DIRS:
                nr, nc = r+dr, c+dc
                nb = (nr, nc)
                if nb in visited: continue
                if not (0<=nr<ROWS and 0<=nc<COLS): continue
                if self.grid[nr][nc] not in PASSABLE: continue
                if self.grid[nr][nc] == FREE and r1<nr<r2 and c1<nc<c2:
                    self.bfs_ops += ops; return nb
                visited.add(nb); queue.append(nb)
        self.bfs_ops += ops; return None

    def _bfs_fallback(self):
        """BFS for any cells the pre-computed sweep plan missed."""
        if not self.free_cells: return None
        start = tuple(self.robot)
        visited = {start}; queue = deque([start]); ops = 0
        while queue:
            r, c = queue.popleft(); ops += 1
            for dr, dc in DIRS:
                nb = (r+dr, c+dc)
                if nb in visited: continue
                nr, nc = nb
                if not (0 <= nr < ROWS and 0 <= nc < COLS): continue
                if self.grid[nr][nc] not in PASSABLE: continue
                if self.grid[nr][nc] == FREE:
                    self.bfs_ops += ops; return nb
                visited.add(nb); queue.append(nb)
        self.bfs_ops += ops; return None

    def _astar(self, start, goal):
        """Weighted A* with closed set.
        Penalises: cleaned cells (revisit), and crossing into a different room
        when both start and goal are in the same room (keeps sweep in-room).
        """
        h      = lambda a,b: abs(a[0]-b[0])+abs(a[1]-b[1])
        g      = {start:0.0}; parent={start:None}
        closed = set(); ctr=0; ops=0
        heap   = [(h(start,goal),ctr,start)]

        start_room = _ROOM_GRID[start[0]][start[1]]
        goal_room  = _ROOM_GRID[goal[0]][goal[1]]
        same_room  = (start_room == goal_room and start_room >= 0)

        while heap:
            f,_,cur = heapq.heappop(heap); ops+=1
            if cur in closed: continue
            closed.add(cur)
            if cur==goal:
                path=[]; node=goal
                while parent[node] is not None:
                    path.append(node); node=parent[node]
                path.reverse(); self.astar_ops+=ops; return path
            for dr,dc in DIRS:
                nb=(cur[0]+dr,cur[1]+dc); nr,nc=nb
                if not (0<=nr<ROWS and 0<=nc<COLS): continue
                if self.grid[nr][nc] not in PASSABLE: continue
                if nb in closed: continue
                step = 1.0
                # Penalise stepping on already-cleaned floor
                if self.grid[nr][nc] == CLEANED:
                    step += REVISIT_PENALTY
                # Penalise leaving the current room when target is in same room
                if same_room and _ROOM_GRID[nr][nc] != start_room:
                    step += 12.0
                ng = g[cur]+step
                if ng<g.get(nb,1e9):
                    g[nb]=ng; parent[nb]=cur; ctr+=1
                    heapq.heappush(heap,(ng+h(nb,goal),ctr,nb))
        self.astar_ops+=ops; return []

    # ═══════════════════════════════════════════════════════════
    #  SENSOR
    # ═══════════════════════════════════════════════════════════
    def _sense(self):
        self.near_obs=set(); r,c=self.robot
        for dr in range(-3,4):
            for dc in range(-3,4):
                nr,nc=r+dr,c+dc
                if 0<=nr<ROWS and 0<=nc<COLS:
                    if self.grid[nr][nc] in (WALL,SOLID_OBS,TABLE_LEG):
                        self.near_obs.add((nr,nc))

    # ═══════════════════════════════════════════════════════════
    #  COVERAGE & CACHE
    # ═══════════════════════════════════════════════════════════
    def _cov(self):
        return self.cleaned_count/max(1,self.total_free)*100

    def _refresh_cache(self):
        if not self._dirty_panel: return
        self._cov_cache = self._cov()
        if self.steps > 0:
            cov = self._cov_cache / 100
            # Cleaning efficiency: only counts energy spent cleaning (fair score)
            self._eff_cache = min(10.0,
                cov * (self.total_free / (self.cleaning_energy + 0.01)) * 10)
            # Overall efficiency: includes return journey energy
            self._overall_eff_cache = min(10.0,
                cov * (self.total_free / (self.energy + 0.01)) * 10)
        else:
            self._overall_eff_cache = 0.0
        self._dirty_panel = False

    def _elapsed_str(self):
        t = self._elapsed
        m = int(t//60); s = int(t%60)
        return f"{m}m {s:02d}s" if m else f"{s}s"

    # ═══════════════════════════════════════════════════════════
    #  MOVE & CLEAN HELPERS
    # ═══════════════════════════════════════════════════════════
    def _mark_clean(self, r, c):
        """Mark a cell cleaned, update all counters and room tracking."""
        if self.grid[r][c] != FREE: return
        self.grid[r][c] = CLEANED
        self.cleaned_count += 1
        self.free_cells.discard((r,c))          # IMP-2
        ri = _ROOM_GRID[r][c]
        if ri >= 0:
            self._room_cleaned[ri] += 1         # IMP-3
            # IMP-8: trigger flash when room hits 100%
            if (not self._room_done[ri] and
                    self._room_cleaned[ri] >= self._room_total[ri]):
                self._room_done[ri] = True
                self._room_flash[ri] = 45       # 45 frames ≈ 0.75 s

    def _move(self, nr, nc):
        d    = (nr-self.robot[0], nc-self.robot[1])
        cost = MOVE_COST+(TURN_COST if self.last_dir and d!=self.last_dir else 0)
        self.energy  += cost
        # Track cleaning vs return energy separately
        if self.phase == "RETURNING":
            self.return_energy += cost
        else:
            self.cleaning_energy += cost
        self.last_dir=d
        self.trail.append(tuple(self.robot))
        self.robot=[nr,nc]; self.steps+=1
        self._sense(); self._dirty_panel=True
        # IMP-4: start timer on first movement
        if self._start_time is None: self._start_time=time.time()

    # ═══════════════════════════════════════════════════════════
    #  SIMULATION STEP
    # ═══════════════════════════════════════════════════════════
    def _step(self):
        if self.finished: return

        # Update elapsed time
        if self._start_time:
            self._elapsed = time.time()-self._start_time

        # Tick room flash counters
        for ri in list(self._room_flash):
            self._room_flash[ri] -= 1
            if self._room_flash[ri] <= 0:
                del self._room_flash[ri]

        if self.phase=="CHARGING": return

        if self.phase=="RETURNING":
            if not self.nav_path:
                self.phase="CHARGING"; self.finished=True
                if self._start_time:
                    self._finish_time=time.time()
                    self._elapsed=self._finish_time-self._start_time
                cov=self._cov()
                clean_eff=min(10.0,(cov/100)*(self.total_free/(self.cleaning_energy+0.01))*10)
                overall_eff=min(10.0,(cov/100)*(self.total_free/(self.energy+0.01))*10)
                self._cov_cache=cov
                self._eff_cache=clean_eff
                self._overall_eff_cache=overall_eff
                g_c,_=_grade(clean_eff); g_o,_=_grade(overall_eff)
                self.msg=(f"Cleaning Completed !!  Simulation Time: {self._elapsed_str()}")
                return
            nr,nc=self.nav_path.pop(0); self._move(nr,nc); return

        if not self.nav_path:
            self.phase  = "BFS"
            target      = self._sweep_target()   # boustrophedon sweep planner
            if target is None:
                self.phase="RETURNING"
                self.nav_path=self._astar(tuple(self.robot),self.start_pos)
                cov=self._cov()
                if not self.nav_path:
                    self.phase="CHARGING"; self.finished=True
                    self.msg=f"Done! Coverage:{cov:.1f}%. Already at dock."
                else:
                    self.msg=f"All clean ({cov:.1f}%)! Returning to dock..."
                return
            self.phase="A*"
            self.nav_path=self._astar(tuple(self.robot),target)
            if not self.nav_path:
                tr,tc=target; self._mark_clean(tr,tc); return

        nr,nc=self.nav_path.pop(0)
        self._mark_clean(nr,nc)
        self._move(nr,nc)

    # ═══════════════════════════════════════════════════════════
    #  IMP-1: STATIC BACKGROUND PRE-RENDERER
    #  Walls + furniture + room labels drawn once; blitted every frame.
    # ═══════════════════════════════════════════════════════════
    def _build_bg(self):
        """Pre-render static elements that never change: walls and doorway tints."""
        surf = pygame.Surface((COLS*CELL, WIN_H))
        surf.fill(BP["floor"])

        # Walls
        for r in range(ROWS):
            for c in range(COLS):
                if self.grid[r][c]==WALL:
                    pygame.draw.rect(surf,BP["wall"],(c*CELL,r*CELL,CELL,CELL))

        # Doorway tint
        for r,c in DOORWAY_CELLS:
            s=pygame.Surface((CELL,CELL),pygame.SRCALPHA)
            s.fill((*BP["door_col"],35))
            surf.blit(s,(c*CELL,r*CELL))

        # Door arcs
        for r,c in[(18,9),(18,22),(18,29),(18,41),(26,17)]:
            x,y=c*CELL+CELL//2,r*CELL+CELL//2
            pygame.draw.arc(surf,BP["door_col"],
                (x-CELL+1,y-CELL+1,CELL*2-2,CELL*2-2),0,math.pi/2,1)

        self._bg_surface=surf

    # ═══════════════════════════════════════════════════════════
    #  DRAW HELPERS (surface-agnostic version for bg pre-render)
    # ═══════════════════════════════════════════════════════════
    def _hatch_on(self, surf, rect, col, step=5):
        x1,y1,w,h=rect
        for i in range(0,w+h,step):
            sx=min(x1+i,x1+w); sy=max(y1,y1+i-w)
            ex=max(x1,x1+i-h); ey=min(y1+h,y1+i)
            if (sx,sy)!=(ex,ey):
                pygame.draw.line(surf,col,(sx,sy),(ex,ey),1)

    def _draw_piece_on(self,surf,r1,c1,r2,c2,label,style):
        px=c1*CELL; py=r1*CELL; pw=(c2-c1+1)*CELL; ph=(r2-r1+1)*CELL; rect=(px,py,pw,ph)
        # All furniture drawn as solid (tables included). Robot passability unchanged.
        pygame.draw.rect(surf,BP["furn_bg"],rect)
        up=label.upper()
        if "SOFA" in up:
            mid=px+pw//2
            pygame.draw.line(surf,BP["furn_ln"],(mid,py+3),(mid,py+ph-3),2)
            for i in range(2):
                cx_=px+3+i*(pw//2-4)
                pygame.draw.rect(surf,BP["hatch"],(cx_,py+3,pw//2-6,ph-6))
                pygame.draw.rect(surf,BP["furn_ln"],(cx_,py+3,pw//2-6,ph-6),1)
        elif "BED" in up:
            pygame.draw.rect(surf,BP["hatch"],(px+4,py+4,pw-8,ph//3))
            pygame.draw.line(surf,BP["furn_ln"],(px,py+ph//3+4),(px+pw,py+ph//3+4),2)
            self._hatch_on(surf,(px+4,py+ph//3+6,pw-8,ph-ph//3-10),BP["hatch"],5)
        elif "WARDROBE" in up or "BOOKSHELF" in up:
            mid=px+pw//2
            pygame.draw.line(surf,BP["furn_ln"],(mid,py+3),(mid,py+ph-3),2)
            pygame.draw.rect(surf,BP["furn_ln"],(px+3,py+3,pw-6,ph-6),1)
        elif "COUNTER" in up:
            for i in range(0,pw,CELL):
                pygame.draw.line(surf,BP["furn_ln"],(px+i,py+3),(px+i,py+ph-3),1)
            pygame.draw.rect(surf,BP["furn_ln"],(px+2,py+2,pw-4,ph-4),2)
        elif "BATHTUB" in up:
            inner=(px+5,py+5,pw-10,ph-10)
            pygame.draw.rect(surf,BP["hatch"],inner)
            pygame.draw.rect(surf,BP["furn_ln"],inner,2)
            pygame.draw.ellipse(surf,BP["furn_bg"],(px+12,py+10,pw-24,ph-20))
            pygame.draw.ellipse(surf,BP["furn_ln"],(px+12,py+10,pw-24,ph-20),1)
        elif "TOILET" in up:
            pygame.draw.rect(surf,BP["hatch"],(px+3,py+3,pw-6,ph//3))
            pygame.draw.ellipse(surf,BP["furn_bg"],(px+4,py+ph//3+2,pw-8,ph*2//3-5))
            pygame.draw.ellipse(surf,BP["furn_ln"],(px+4,py+ph//3+2,pw-8,ph*2//3-5),2)
        elif "SINK" in up:
            pygame.draw.ellipse(surf,BP["hatch"],(px+5,py+5,pw-10,ph-10))
            pygame.draw.ellipse(surf,BP["furn_ln"],(px+5,py+5,pw-10,ph-10),2)
            pygame.draw.circle(surf,BP["furn_ln"],(px+pw//2,py+ph//2),3,2)
        elif "TV" in up:
            pygame.draw.rect(surf,(32,48,98),(px+4,py+4,pw-8,ph-8))
            pygame.draw.rect(surf,BP["furn_bg"],(px+8,py+8,pw-16,ph-16))
        elif "CHAIR" in up:
            pygame.draw.rect(surf,BP["furn_ln"],(px+3,py+3,pw-6,ph//2-2),2)
            self._hatch_on(surf,(px+3,py+ph//2+1,pw-6,ph//2-4),BP["hatch"],4)
        elif "PLANT" in up:
            cx_=px+pw//2; cy_=py+ph//2; r_=min(pw,ph)//2-2
            pygame.draw.circle(surf,BP["furn_ln"],(cx_,cy_),r_,2)
            pygame.draw.line(surf,BP["furn_ln"],(cx_-r_+2,cy_),(cx_+r_-2,cy_),1)
            pygame.draw.line(surf,BP["furn_ln"],(cx_,cy_-r_+2),(cx_,cy_+r_-2),1)
        elif "DINING" in up:
            # Dining table — hatched wooden top with corner legs
            self._hatch_on(surf,(px+4,py+4,pw-8,ph-8),BP["hatch"],6)
            pygame.draw.rect(surf,BP["furn_ln"],(px+3,py+3,pw-6,ph-6),2)
            for lr_,lc_ in((r1,c1),(r1,c2),(r2,c1),(r2,c2)):
                lx_=lc_*CELL+CELL//2; ly_=lr_*CELL+CELL//2
                pygame.draw.circle(surf,BP["leg"],(lx_,ly_),5)
                pygame.draw.circle(surf,BP["furn_ln"],(lx_,ly_),5,1)
        elif "COFFEE" in up:
            # Coffee table — glass-top look
            pygame.draw.rect(surf,BP["table_top"],(px+3,py+3,pw-6,ph-6))
            pygame.draw.line(surf,BP["table_ln"],(px+3,py+3),(px+pw-3,py+ph-3),1)
            pygame.draw.line(surf,BP["table_ln"],(px+pw-3,py+3),(px+3,py+ph-3),1)
            pygame.draw.rect(surf,BP["furn_ln"],(px+3,py+3,pw-6,ph-6),2)
            for lr_,lc_ in((r1,c1),(r1,c2),(r2,c1),(r2,c2)):
                lx_=lc_*CELL+CELL//2; ly_=lr_*CELL+CELL//2
                pygame.draw.circle(surf,BP["leg"],(lx_,ly_),4)
                pygame.draw.circle(surf,BP["furn_ln"],(lx_,ly_),4,1)
        elif "SIDE" in up:
            # Side table — compact hatched top
            self._hatch_on(surf,(px+3,py+3,pw-6,ph-6),BP["hatch"],4)
            pygame.draw.rect(surf,BP["furn_ln"],(px+3,py+3,pw-6,ph-6),2)
            for lr_,lc_ in((r1,c1),(r1,c2),(r2,c1),(r2,c2)):
                lx_=lc_*CELL+CELL//2; ly_=lr_*CELL+CELL//2
                pygame.draw.circle(surf,BP["leg"],(lx_,ly_),3)
                pygame.draw.circle(surf,BP["furn_ln"],(lx_,ly_),3,1)
        elif "ISL" in up:
            # Kitchen island — counter-style with segment lines
            for i in range(0,pw,CELL):
                pygame.draw.line(surf,BP["furn_ln"],(px+i,py+3),(px+i,py+ph-3),1)
            pygame.draw.rect(surf,BP["furn_ln"],(px+2,py+2,pw-4,ph-4),2)
            for lr_,lc_ in((r1,c1),(r1,c2),(r2,c1),(r2,c2)):
                lx_=lc_*CELL+CELL//2; ly_=lr_*CELL+CELL//2
                pygame.draw.circle(surf,BP["leg"],(lx_,ly_),4)
                pygame.draw.circle(surf,BP["furn_ln"],(lx_,ly_),4,1)
        elif "DESK" in up:
            self._hatch_on(surf,(px+3,py+3,pw-6,ph-6),BP["hatch"],5)
            pygame.draw.rect(surf,BP["furn_ln"],(px+2,py+2,pw-4,ph-4),2)
        elif "FRIDGE" in up:
            pygame.draw.line(surf,BP["furn_ln"],(px+3,py+ph//3+3),(px+pw-3,py+ph//3+3),2)
            pygame.draw.rect(surf,BP["furn_ln"],(px+3,py+3,pw-6,ph-6),1)
        elif "STOVE" in up:
            for ri in range(2):
                for ci in range(2):
                    ecx=px+4+ci*(pw//2-3); ecy=py+5+ri*(ph//2-3)
                    pygame.draw.circle(surf,BP["hatch"],(ecx,ecy),4)
                    pygame.draw.circle(surf,BP["furn_ln"],(ecx,ecy),4,1)
        elif "NIGHT" in up:
            pygame.draw.rect(surf,BP["furn_ln"],(px+3,py+3,pw-6,ph-6),1)
            pygame.draw.circle(surf,BP["hatch"],(px+pw//2,py+ph//2),min(pw,ph)//4)
        elif "CABINET" in up:
            mid=px+pw//2
            pygame.draw.line(surf,BP["furn_ln"],(mid,py+3),(mid,py+ph-3),2)
            pygame.draw.rect(surf,BP["furn_ln"],(px+3,py+3,pw-6,ph-6),1)
        else:
            self._hatch_on(surf,(px+3,py+3,pw-6,ph-6),BP["hatch"],5)
        pygame.draw.rect(surf,BP["furn_ln"],rect,2)
        if pw>22 and ph>14:
            words=label.split(); l1=words[0]; l2=words[1] if len(words)>1 else ""
            s1=self.fxs.render(l1,True,BP["furn_txt"])
            lx=px+pw//2-s1.get_width()//2; ly=py+ph//2-s1.get_height()-(1 if l2 else 0)
            surf.blit(s1,(lx,ly))
            if l2:
                s2=self.fxs.render(l2,True,BP["furn_txt"])
                surf.blit(s2,(px+pw//2-s2.get_width()//2,ly+s1.get_height()+1))

    # ═══════════════════════════════════════════════════════════
    #  CHARGING DOCK DRAW
    # ═══════════════════════════════════════════════════════════
    def _draw_charging_dock(self):
        sr,sc=self.start_pos
        cx=sc*CELL+CELL//2; cy=sr*CELL+CELL//2
        pulse=abs(math.sin(self.charge_anim*0.08)) if self.phase=="CHARGING" else 0.7
        rc=tuple(int(a*pulse+30*(1-pulse)) for a in BP["charge_ring"])
        pygame.draw.circle(self.screen,rc,(cx,cy),CELL//2,3)
        pygame.draw.circle(self.screen,BP["charge_bg"],(cx,cy),CELL//2-4)
        bx,by=cx-3,cy-6
        bolt=[(bx+4,by),(bx+1,by+5),(bx+4,by+4),(bx,by+10),(bx+5,by+5),(bx+2,by+5)]
        pygame.draw.polygon(self.screen,BP["charge_bolt"],bolt)
        lbl=self.fxs.render("DOCK",True,BP["charge_ring"])
        self.screen.blit(lbl,(cx-lbl.get_width()//2,cy+CELL//2+2))

    # ═══════════════════════════════════════════════════════════
    #  MAIN GRID DRAW
    # ═══════════════════════════════════════════════════════════
    def _draw_grid(self):
        # Layer 1: static background (walls + doorways only)
        if self._bg_surface is None: self._build_bg()
        self.screen.blit(self._bg_surface,(0,0))

        # Layer 2: dynamic floor — every passable cell incl. table interiors
        for r in range(ROWS):
            for c in range(COLS):
                x,y=c*CELL,r*CELL; cell=self.grid[r][c]
                if cell==WALL or cell in(SOLID_OBS,TABLE_LEG): continue
                col=BP["floor_clean"] if cell==CLEANED else BP["floor"]
                pygame.draw.rect(self.screen,col,(x,y,CELL,CELL))
                pygame.draw.rect(self.screen,BP["grid"],(x,y,CELL,CELL),1)

        # Layer 3: furniture drawn OVER floor (tables now opaque & labelled)
        drawn=set()
        for label,r1,c1,r2,c2,style in FURNITURE_DEFS:
            key=(label,r1,c1)
            if key in drawn: continue
            drawn.add(key)
            self._draw_piece_on(self.screen,r1,c1,r2,c2,label,style)

        # Layer 4: room name labels — positioned at black-line locations from user
        ROOM_LABEL_POS = {
            "LIVING ROOM":     (9,  13),  # up 3 from 12
            "MASTER BEDROOM":  (10, 37),  # up 4 from 14
            "KITCHEN":         (24, 13),  # up 4 from 28
            "DINING ROOM":     (30, 25),  # down 3 from 27
            "BATHROOM":        (23, 43),  # up 8 from 31
        }
        for label,r1,c1,r2,c2 in ROOMS:
            lrow, lcol = ROOM_LABEL_POS.get(label, ((r1+r2)//2, (c1+c2)//2))
            cx = lcol*CELL + CELL//2
            cy = lrow*CELL + CELL//2
            ls = self.fs.render(label, True, (0, 0, 0))
            self.screen.blit(ls, (cx-ls.get_width()//2, cy-ls.get_height()//2))

        # Layer 5: room completion flash
        for ri,frames in self._room_flash.items():
            _,r1,c1,r2,c2=ROOMS[ri]
            alpha=int(120*(frames/45))
            s=pygame.Surface(((c2-c1+1)*CELL,(r2-r1+1)*CELL),pygame.SRCALPHA)
            s.fill((*BP["green_flash"],alpha))
            self.screen.blit(s,(c1*CELL,r1*CELL))

        # Layer 6: LIDAR sensor glow
        for nr,nc in self.near_obs:
            x,y=nc*CELL,nr*CELL
            s=pygame.Surface((CELL,CELL),pygame.SRCALPHA)
            s.fill((*BP["sensor"],65)); self.screen.blit(s,(x,y))
            pygame.draw.rect(self.screen,BP["sensor"],(x,y,CELL,CELL),1)

        # Layer 7: charging dock
        self._draw_charging_dock()

        # Layer 8: trail
        for tr,tc in self.trail:
            pygame.draw.circle(self.screen,BP["trail"],
                (tc*CELL+CELL//2,tr*CELL+CELL//2),2)

        # Layer 9: A* planned path
        for pr,pc in self.nav_path:
            pcol=BP["return_col"] if self.phase=="RETURNING" else BP["nav_path"]
            pygame.draw.circle(self.screen,pcol,
                (pc*CELL+CELL//2,pr*CELL+CELL//2),3)

        # Layer 10: robot (always on top)
        rx=self.robot[1]*CELL+CELL//2; ry=self.robot[0]*CELL+CELL//2; rad=CELL//2-2
        rcol=(BP["charge_bg"] if self.phase=="CHARGING" else
              BP["return_col"] if self.phase=="RETURNING" else BP["robot"])
        glow=pygame.Surface((CELL*2,CELL*2),pygame.SRCALPHA)
        pygame.draw.circle(glow,(*rcol,55),(CELL,CELL),CELL-2)
        self.screen.blit(glow,(rx-CELL,ry-CELL))
        pygame.draw.circle(self.screen,rcol,(rx,ry),rad)
        pygame.draw.circle(self.screen,BP["robot_ring"],(rx,ry),rad,2)
        if self.last_dir and self.phase!="CHARGING":
            ex=rx+self.last_dir[1]*(rad-4); ey=ry+self.last_dir[0]*(rad-4)
            pygame.draw.circle(self.screen,BP["white"],(ex,ey),4)
            pygame.draw.circle(self.screen,BP["robot_eye"],(ex,ey),2)

    # ═══════════════════════════════════════════════════════════
    #  PANEL
    # ═══════════════════════════════════════════════════════════
    def _bar(self,x,y,w,h,pct,col):
        pygame.draw.rect(self.screen,BP["bar_bg"],(x,y,w,h),border_radius=4)
        f=max(0,int(w*min(1,pct)))
        if f: pygame.draw.rect(self.screen,col,(x,y,f,h),border_radius=4)

    def _t(self,txt,font,col,x,y):
        s=font.render(txt,True,col); self.screen.blit(s,(x,y)); return s.get_height()

    def _section(self, title, px, x, pw, y, col=(92,185,255)):
        """Draw a pill-style section header and return new y."""
        pygame.draw.rect(self.screen,(22,34,68),(x,y,pw,18),border_radius=4)
        pygame.draw.line(self.screen,col,(x+6,y+9),(x+10,y+9),2)
        s=self.fxs.render(title.upper(),True,col)
        self.screen.blit(s,(x+16,y+2))
        return y+22

    def _stat_card(self, label, value, x, y, w, h, bar_pct, bar_col, val_col):
        """Draw a labelled stat card with inline progress bar."""
        pygame.draw.rect(self.screen,(22,34,68),(x,y,w,h),border_radius=5)
        pygame.draw.rect(self.screen,(40,58,110),(x,y,w,h),1,border_radius=5)
        lbl_s=self.fxs.render(label,True,BP["txt_dim"])
        val_s=self.fs.render(value,True,val_col)
        self.screen.blit(lbl_s,(x+7,y+4))
        self.screen.blit(val_s,(x+w-val_s.get_width()-7,y+4))
        # thin bar at bottom of card
        bar_x,bar_y=x+6,y+h-6
        bar_w=w-12
        pygame.draw.rect(self.screen,(38,52,98),(bar_x,bar_y,bar_w,3),border_radius=2)
        f=max(0,int(bar_w*min(1,bar_pct)))
        if f: pygame.draw.rect(self.screen,bar_col,(bar_x,bar_y,f,3),border_radius=2)

    def _draw_panel(self):
        self._refresh_cache()
        cov=self._cov_cache; eff=self._eff_cache
        grade_ltr,grade_col=_grade(eff)

        px=COLS*CELL
        # Panel background with subtle gradient feel via two rects
        pygame.draw.rect(self.screen,(12,18,42),(px,0,PANEL_W,WIN_H))
        pygame.draw.rect(self.screen,(18,26,56),(px,0,PANEL_W,WIN_H//2))
        # Left border accent line
        pygame.draw.line(self.screen,(55,90,180),(px,0),(px,WIN_H),2)

        x=px+10; pw=PANEL_W-20; y=8

        # ═══════════════════════════════════════════════════
        # ALGORITHM STATUS
        # ═══════════════════════════════════════════════════
        y=self._section("Algorithms",px,x,pw,y)

        half=(pw-4)//2
        # BFS card
        ab=(self.phase=="BFS" and self.running)
        bcol=BP["bfs_col"] if ab else (40,58,100)
        pygame.draw.rect(self.screen,(22,38,72) if ab else(18,28,55),(x,y,half,52),border_radius=5)
        pygame.draw.rect(self.screen,bcol,(x,y,half,52),2 if ab else 1,border_radius=5)
        self._t("BFS",self.fm,BP["bfs_col"],x+6,y+5)
        self._t("Coverage",self.fxs,BP["txt_dim"],x+6,y+22)
        self._t("Planner",self.fxs,BP["txt_dim"],x+6,y+32)
        if ab:
            dot_r=4
            for di,dcol in enumerate([(BP["bfs_col"],180),(BP["bfs_col"],100),(BP["bfs_col"],40)]):
                alpha=int(abs(math.sin((self.anim_tick*0.12)+di*1.0))*dcol[1])
                ds=pygame.Surface((dot_r*2,dot_r*2),pygame.SRCALPHA)
                pygame.draw.circle(ds,(*BP["bfs_col"],alpha),(dot_r,dot_r),dot_r)
                self.screen.blit(ds,(x+half-14+di*9,y+38))
        y_bfs=y

        # A* card
        aa=(self.phase=="A*" and self.running)
        acol=BP["astar_col"] if aa else (40,58,100)
        ax=x+half+4
        pygame.draw.rect(self.screen,(50,38,12) if aa else(18,28,55),(ax,y_bfs,half,52),border_radius=5)
        pygame.draw.rect(self.screen,acol,(ax,y_bfs,half,52),2 if aa else 1,border_radius=5)
        self._t("A*",self.fm,BP["astar_col"],ax+6,y_bfs+5)
        self._t("Weighted",self.fxs,BP["txt_dim"],ax+6,y_bfs+22)
        self._t("Navigator",self.fxs,BP["txt_dim"],ax+6,y_bfs+32)
        y+=56

        # Active phase pill
        phase_bg  = {
            "BFS":      ((20,50,80),  BP["bfs_col"],   "BFS  Sweeping: " + (self._sweep_room_names[self._sweep_idx] if self._sweep_idx < len(self._sweep_room_names) else "done")),
            "A*":       ((50,38,10),  BP["astar_col"], "A*   Navigating to target..."),
            "RETURNING":((55,30,10),  BP["return_col"],"↩   Returning to dock..."),
            "CHARGING": ((40,38,5),   BP["charge_bg"], "⚡  Charging at dock"),
        }
        if self.phase in phase_bg:
            pbg,pcol,ptxt=phase_bg[self.phase]
            if self.phase=="CHARGING":
                pv=int(150+105*abs(math.sin(self.charge_anim*0.08)))
                pcol=(pv,int(pv*0.85),0)
            pygame.draw.rect(self.screen,pbg,(x,y,pw,18),border_radius=9)
            pygame.draw.rect(self.screen,pcol,(x,y,pw,18),1,border_radius=9)
            ps=self.fxs.render(ptxt,True,pcol)
            self.screen.blit(ps,(x+pw//2-ps.get_width()//2,y+2))
            y+=22
        y+=4

        # ═══════════════════════════════════════════════════
        # LIVE STATISTICS — 2×2 stat cards
        # ═══════════════════════════════════════════════════
        y=self._section("Statistics",px,x,pw,y)

        me=self.total_free*(MOVE_COST+TURN_COST); ep=self.energy/max(1,me)
        ec=BP["bar_e_lo"] if ep<0.5 else BP["bar_e_mid"] if ep<0.8 else BP["bar_e_hi"]
        cov_col=(100,220,120) if cov>=95 else (75,195,255) if cov>=60 else (235,145,35)
        cw=(pw-4)//2; ch=36

        # Row 1: Coverage | Energy
        self._stat_card("COVERAGE",  f"{cov:.1f}%",   x,    y, cw, ch, cov/100,  BP["bar_cov"], cov_col)
        self._stat_card("ENERGY",    f"{self.energy:.0f}u", x+cw+4,y,cw,ch,ep,ec,ec)
        y+=ch+4
        # Row 2: Steps | Total Layout Area
        self._stat_card("STEPS", str(self.steps), x, y, cw, ch, min(self.steps/3000,1),(100,160,255),(160,200,255))
        _total_cells_stat = sum((r2-r1+1)*(c2-c1+1) for _,r1,c1,r2,c2 in ROOMS)
        _total_sqft_stat  = _total_cells_stat * 1
        self._stat_card("LAYOUT", f"{_total_sqft_stat:.0f} sqft", x+cw+4, y, cw, ch, 1.0, (100,160,255),(160,200,255))
        y+=ch+4

        # Efficiency + grade (full-width card) — two scores shown
        if self.steps>0:
            overall_eff = self._overall_eff_cache
            o_grade_ltr, o_grade_col = _grade(overall_eff)

            # ── Cleaning efficiency card (primary score) ──────
            pygame.draw.rect(self.screen,(22,34,68),(x,y,pw,56),border_radius=5)
            pygame.draw.rect(self.screen,grade_col,(x,y,pw,56),2,border_radius=5)
            # grade badge
            pygame.draw.rect(self.screen,(30,40,80),(x+4,y+4,32,48),border_radius=4)
            gs=self.fm.render(grade_ltr,True,grade_col)
            self.screen.blit(gs,(x+4+16-gs.get_width()//2,y+4+24-gs.get_height()//2))
            # labels
            grade_desc={"S":"Outstanding","A":"Excellent","B":"Good","C":"Average","D":"Needs work"}
            self._t("CLEANING EFFICIENCY",self.fxs,(160,178,220),x+42,y+5)
            self._t(f"{eff:.2f}/10  {grade_desc.get(grade_ltr,'')}",self.fs,grade_col,x+42,y+18)
            self._t("(return journey excluded)",self.fxs,(100,118,165),x+42,y+32)
            # bar
            bx2,bw2=x+42,pw-48
            pygame.draw.rect(self.screen,(38,52,98),(bx2,y+46,bw2,4),border_radius=2)
            f2=max(0,int(bw2*eff/10))
            if f2: pygame.draw.rect(self.screen,grade_col,(bx2,y+46,f2,4),border_radius=2)
            y+=62

            # ── Overall efficiency card (includes return) ─────
            pygame.draw.rect(self.screen,(18,26,52),(x,y,pw,52),border_radius=5)
            pygame.draw.rect(self.screen,o_grade_col,(x,y,pw,52),2,border_radius=5)
            pygame.draw.rect(self.screen,(28,36,70),(x+4,y+4,28,44),border_radius=4)
            og=self.fs.render(o_grade_ltr,True,o_grade_col)
            self.screen.blit(og,(x+4+14-og.get_width()//2,y+4+22-og.get_height()//2))
            self._t("OVERALL EFFICIENCY",self.fxs,(130,148,190),x+38,y+5)
            self._t(f"{overall_eff:.2f}/10  {grade_desc.get(o_grade_ltr,'')}",
                    self.fs,o_grade_col,x+38,y+18)
            self._t(f"Return: {self.return_energy:.0f}u   Cleaning: {self.cleaning_energy:.0f}u",
                    self.fxs,(90,108,150),x+38,y+32)
            # bar
            pygame.draw.rect(self.screen,(38,52,98),(x+38,y+44,pw-44,4),border_radius=2)
            f3=max(0,int((pw-44)*overall_eff/10))
            if f3: pygame.draw.rect(self.screen,o_grade_col,(x+38,y+44,f3,4),border_radius=2)
            y+=58
        y+=4

        # ═══════════════════════════════════════════════════
        # ROOM PROGRESS
        # ═══════════════════════════════════════════════════
        y=self._section("Room Progress",px,x,pw,y)

        room_short=["Living","Bedroom","Kitchen","Dining","Bath"]
        for i,(rname,r1,c1,r2,c2) in enumerate(ROOMS):
            total  =self._room_total[i]; cleaned=self._room_cleaned[i]
            pct    =cleaned/max(1,total); done=self._room_done[i]
            lcol   =(100,220,120) if done else BP["txt_dim"]
            bcol_r =BP["bar_cov"] if done else (55,125,215)
            room_sqft = (r2-r1+1)*(c2-c1+1)*1
            # row bg
            pygame.draw.rect(self.screen,(20,30,60),(x,y,pw,26),border_radius=3)
            # tick or dot
            if done:
                pygame.draw.circle(self.screen,(80,200,100),(x+8,y+13),4)
            else:
                pygame.draw.circle(self.screen,(55,72,128),(x+8,y+13),4,1)
            # name
            ns=self.fxs.render(room_short[i],True,lcol)
            self.screen.blit(ns,(x+18,y+2))
            # sqft
            sq=self.fxs.render(f"{room_sqft:.0f} sqft",True,(110,130,175))
            self.screen.blit(sq,(x+18,y+13))
            # bar
            bx3=x+72; bw3=pw-100
            pygame.draw.rect(self.screen,(38,52,98),(bx3,y+10,bw3,6),border_radius=3)
            f3=max(0,int(bw3*pct))
            if f3: pygame.draw.rect(self.screen,bcol_r,(bx3,y+10,f3,6),border_radius=3)
            # pct
            ps2=self.fxs.render(f"{pct*100:.0f}%",True,lcol)
            self.screen.blit(ps2,(x+pw-ps2.get_width(),y+2))
            y+=29
        y+=4

        # ═══════════════════════════════════════════════════
        # LEGEND — 2-column grid
        # ═══════════════════════════════════════════════════
        y=self._section("Legend",px,x,pw,y)

        items=[
            (BP["floor_clean"], "Cleaned"),
            (BP["floor"],       "Uncleaned"),
            (BP["wall"],        "Wall"),
            (BP["furn_bg"],     "Furniture"),
            (BP["table_top"],   "Table"),
            (BP["door_col"],    "Doorway"),
            (BP["charge_bg"],   "Dock"),
            (BP["robot"],       "Robot"),
            (BP["return_col"],  "Returning"),
            (BP["trail"],       "Trail"),
            (BP["nav_path"],    "A* Path"),
            (BP["sensor"],      "LIDAR"),
        ]
        col_w=pw//2
        for idx,(ic,lbl) in enumerate(items):
            cx2=x+(idx%2)*col_w; cy2=y+(idx//2)*15
            pygame.draw.rect(self.screen,ic,(cx2,cy2+2,10,10),border_radius=2)
            pygame.draw.rect(self.screen,(55,72,128),(cx2,cy2+2,10,10),1,border_radius=2)
            ls=self.fxs.render(lbl,True,BP["txt_dim"])
            self.screen.blit(ls,(cx2+13,cy2+1))
        y+=(len(items)//2)*15+8



        # ═══════════════════════════════════════════════════
        # CONTROLS — compact 2-column grid
        # ═══════════════════════════════════════════════════
        y=self._section("Controls",px,x,pw,y)

        ctrls=[("SPC","Start/Pause"),("R","Reset"),
               ("+/-","Speed"),("S","Single step"),
               (">/<","Fast / Slow"),("Q","Quit")]
        col_w2=pw//2
        for idx,(key,desc) in enumerate(ctrls):
            cx3=x+(idx%2)*col_w2; cy3=y+(idx//2)*17
            kw=self.fxs.size(key)[0]
            pygame.draw.rect(self.screen,(45,58,105),(cx3,cy3+1,kw+8,13),border_radius=3)
            ks=self.fxs.render(key,True,(220,235,255))
            self.screen.blit(ks,(cx3+4,cy3+2))
            ds=self.fxs.render(desc,True,BP["txt_dim"])
            self.screen.blit(ds,(cx3+kw+12,cy3+2))
        y+=(len(ctrls)//2)*17+5

        # ═══════════════════════════════════════════════════
        # STATUS BAR at bottom
        # ═══════════════════════════════════════════════════
        status_h=max(WIN_H-y-4, 22)
        sc_bg=((20,55,30) if self.finished else
               (40,36,10) if self.running  else(20,24,50))
        sc=((65,232,115) if self.finished else
            (225,190,45)  if self.running  else(120,140,185))
        pygame.draw.rect(self.screen,sc_bg,(x,y,pw,status_h),border_radius=4)
        pygame.draw.rect(self.screen,sc,(x,y,pw,status_h),1,border_radius=4)
        words,line=[],""
        sy=y+4
        for w in self.msg.split():
            t=line+w+" "
            if self.fxs.size(t)[0]>pw-10:
                ss=self.fxs.render(line.strip(),True,sc)
                self.screen.blit(ss,(x+5,sy)); sy+=13; line=w+" "
            else: line=t
        if line:
            ss=self.fxs.render(line.strip(),True,sc)
            self.screen.blit(ss,(x+5,sy))

    # ═══════════════════════════════════════════════════════════
    #  MAIN LOOP
    # ═══════════════════════════════════════════════════════════
    def run(self):
        while True:
            self.clock.tick(FPS)

            for ev in pygame.event.get():
                if ev.type==pygame.QUIT: pygame.quit(); sys.exit()
                if ev.type==pygame.KEYDOWN:
                    if ev.key==pygame.K_SPACE:
                        if self.finished: self.reset()
                        else:
                            self.running=not self.running
                            self.msg=("Running — same-room BFS + weighted A*..."
                                      if self.running else "Paused.  SPACE to resume.")
                    elif ev.key==pygame.K_r: self.reset()
                    # IMP-6: S = single step
                    elif ev.key==pygame.K_s:
                        if not self.finished:
                            self.running=False; self._step()
                            self.msg="Step mode — press S to advance one step."
                    # Speed up / down
                    elif ev.key in(pygame.K_EQUALS,pygame.K_PLUS,pygame.K_KP_PLUS):
                        self.move_delay=max(1,self.move_delay-1)
                        self.msg=f"Speed up  delay={self.move_delay} steps_tick={self.steps_tick}"
                    elif ev.key in(pygame.K_MINUS,pygame.K_KP_MINUS):
                        self.move_delay=min(30,self.move_delay+1)
                        self.msg=f"Speed down  delay={self.move_delay}"
                    # IMP-7: > = fast-forward (more steps per tick)
                    elif ev.key in(pygame.K_PERIOD,pygame.K_GREATER):
                        self.steps_tick=min(20,self.steps_tick+1)
                        self.msg=f"Fast-forward  {self.steps_tick} steps/tick"
                    elif ev.key in(pygame.K_COMMA,pygame.K_LESS):
                        self.steps_tick=max(1,self.steps_tick-1)
                        self.msg=f"Slow-forward  {self.steps_tick} steps/tick"
                    elif ev.key==pygame.K_q:
                        pygame.quit()
                        return

            if self.running and not self.finished:
                self.mtimer+=1
                if self.mtimer>=self.move_delay:
                    self.mtimer=0
                    for _ in range(self.steps_tick):  # IMP-7
                        if not self.finished: self._step()

            if self.phase=="CHARGING":
                self.charge_anim+=1
            self.anim_tick+=1   # increments every frame for animations

            # Update elapsed while running
            if self._start_time and not self.finished:
                self._elapsed=time.time()-self._start_time

            self.screen.fill(BP["floor"])
            self._draw_grid()
            self._draw_panel()
            pygame.display.flip()

if __name__=="__main__":
    Sim().run()
