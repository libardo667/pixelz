"""
Dwarf‑like Pixel World Simulation with GUI and AI log stubs.

This module implements a more complex simulation inspired by games like
Dwarf Fortress, combining cellular resources, moving characters and
per‑character log generation.  The simulation consists of a 2D grid of
tiles containing resources (trees, stone, water, empty) and a small
number of character entities.  Characters move to satisfy their
desires, harvest resources, age, reproduce and eventually die.

Key features:

* **Resource tiles** – The world spawns different tile types (trees,
  stone, water and empty).  Resources can be harvested by characters.
* **Characters** – Only a handful of characters are placed at the
  beginning.  Each has a name, location, hunger level, age and
  internal memory of states.  They move around the grid, harvest
  resources, reproduce after reaching a certain age and die after
  exceeding a maximum lifespan.
* **Memory and logging** – Every simulation step records the
  character's state into a JSON file.  Memory length grows as the
  character ages then gradually declines.  When the player clicks on a
  character tile, the stored memory is summarised into a human‑readable
  log entry (simulating an AI API call).  This log is appended to the
  JSON file for later review.
* **GUI** – The GUI uses ``tkinter``.  The grid is rendered on a
  canvas with colours representing tile types and characters.  Clicking
  a character tile pops up a window with its AI log.  The simulation
  runs automatically and can be extended with buttons for pausing or
  stepping if desired.

Usage:

    python dwarf_world_gui.py

This will open a window showing the simulation.  Note that the ChatGPT
environment cannot display GUI windows; run this locally on your
machine instead.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
import tkinter as tk
from typing import Dict, List, Optional, Tuple


# =============================================================================
# Configuration constants
# =============================================================================

WORLD_WIDTH = 200
WORLD_HEIGHT = 150
CELL_SIZE = 10  # pixels per tile in GUI
# Default time between simulation steps (ms).  Increase this to slow down
# movement and make clicking easier.  Users can change this constant as desired.
UPDATE_INTERVAL_MS = 1000

# Resource distribution probabilities
RESOURCE_PROBABILITIES: Dict[str, float] = {
    "tree": 0.08,
    "plant": 0.05,
    "stone": 0.05,
    "water": 0.05,
    "empty": 0.77,
}

# Colour mapping for resources and characters
TILE_COLOURS: Dict[str, str] = {
    "tree": "#228B22",   # forest green
    "plant": "#32CD32",   # lime green for plants
    "stone": "#696969",  # dim grey
    "water": "#1E90FF",  # dodger blue
    "empty": "#F5DEB3",  # wheat
    "stockpile": "#b39ddb",  # violet for stockpile tiles
    "character": "#8B4513",  # saddle brown
}

# Character settings
INITIAL_CHARACTER_COUNT = 10
NAME_POOL = [
    "Aldar", "Belin", "Cyra", "Doran", "Elaia", "Finn", "Galen", "Helia",
    "Ivor", "Jora", "Kael", "Lira", "Milo", "Nera", "Orrin", "Pella",
    "Quin", "Rhea", "Soren", "Talia", "Ulric", "Vera", "Wyn", "Xan",
    "Yara", "Zane",
]
REPRODUCTION_AGE = 20
REPRODUCTION_PROBABILITY = 0.1
DEATH_AGE = 100
HUNGER_THRESHOLD = 20

# Threshold for thirst (analogous to hunger)
THIRST_THRESHOLD = 20

# Steps required for resources to regrow after harvesting or drinking
REGROW_STEPS: Dict[str, int] = {
    "plant": 20,
    "water": 30,
    "stone": 40,
    "tree": 50,
}

# Resource types that can be stockpiled. Each has its own stockpile tile.
STOCKPILE_RESOURCES: List[str] = ["tree", "plant", "stone"]

# Memory configuration
BASE_MEMORY_LENGTH = 5
MAX_MEMORY_LENGTH = 30

STATE_DIR = Path("character_states")


# =============================================================================
# Helper functions
# =============================================================================

def choose_resource() -> str:
    """Randomly choose a resource tile based on predefined probabilities."""
    rnd = random.random()
    cumulative = 0.0
    for resource, prob in RESOURCE_PROBABILITIES.items():
        cumulative += prob
        if rnd < cumulative:
            return resource
    return "empty"  # fallback


def generate_name() -> str:
    """Select a random name from the pool."""
    return random.choice(NAME_POOL)


def ensure_state_dir() -> None:
    """Ensure the directory for character JSON files exists."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Data classes
# =============================================================================

class Character:
    """
    Represents a character in the simulation.  Characters occupy a tile
    in the world, move towards resources, harvest, age, reproduce and
    die.  Each character maintains a memory (list of state snapshots)
    that is written to a JSON file every step.  The file also stores
    AI‑generated logs (simulated here).
    """

    _next_id = 0

    def __init__(self, name: str, x: int, y: int):
        self.id: int = Character._next_id
        Character._next_id += 1
        self.name: str = name
        self.x: int = x
        self.y: int = y
        self.age: int = 0
        self.hunger: int = random.randint(80, 100)
        # Initialise thirst in the same range as hunger.  Characters will seek water when thirsty.
        self.thirst: int = random.randint(80, 100)
        # ``horny`` indicates readiness to reproduce.  It is initially set based on
        # being satiated at construction but will be updated each step in
        # ``World.character_step``.  A character can only be horny when fully
        # satiated (hunger and thirst above their thresholds) and not carrying
        # any resource.  The initial value uses a 50% chance for variety but
        # does not otherwise influence behaviour.
        self.horny: bool = self.age >= REPRODUCTION_AGE and self.hunger > HUNGER_THRESHOLD and self.thirst > THIRST_THRESHOLD and random.random() < 0.5
        # Resource currently being carried for deposit.  ``None`` means not carrying anything.
        self.carrying: Optional[str] = None
        # When satiated, characters pick a resource type to stockpile.  ``None`` indicates no current target.
        self.stockpile_target: Optional[str] = None
        self.memory: List[Dict[str, int | str]] = []
        self.ai_logs: List[str] = []
        # Track mates and the number of times reproduced with each.  Key is partner id.
        self.mates: Dict[int, int] = {}

    def state_snapshot(self) -> Dict[str, int | str]:
        """Return a dict representing the current state."""
        return {
            "step": self.age,
            "name": self.name,
            "x": self.x,
            "y": self.y,
            "hunger": self.hunger,
            "thirst": self.thirst,
            # Store what the character is currently carrying (empty string if none)
            "carrying": self.carrying or "",
        }

    def memory_length(self) -> int:
        """
        Calculate the current memory length based on age.  Memory grows
        up to half of the lifespan then gradually declines.
        """
        half = DEATH_AGE // 2
        if self.age <= half:
            return min(BASE_MEMORY_LENGTH + self.age // 2, MAX_MEMORY_LENGTH)
        else:
            decline = self.age - half
            return max(BASE_MEMORY_LENGTH, MAX_MEMORY_LENGTH - decline)

    def update_memory(self) -> None:
        """Trim memory to the appropriate length based on current age."""
        limit = self.memory_length()
        if len(self.memory) > limit:
            self.memory = self.memory[-limit:]

    def save_state(self) -> None:
        """Write the character's memory and AI logs to a JSON file."""
        ensure_state_dir()
        data = {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "hunger": self.hunger,
            "thirst": self.thirst,
            "memory": self.memory,
            "ai_logs": self.ai_logs,
        }
        path = STATE_DIR / f"character_{self.id}.json"
        path.write_text(json.dumps(data, indent=2))

    def load_state(self) -> None:
        """Load existing memory and AI logs from disk if present."""
        path = STATE_DIR / f"character_{self.id}.json"
        if path.is_file():
            data = json.loads(path.read_text())
            self.memory = data.get("memory", [])
            self.ai_logs = data.get("ai_logs", [])
            # Restore thirst if persisted; if not present, initialise randomly
            self.thirst = data.get("thirst", random.randint(5, 20))

# =============================================================================
# World and simulation logic
# =============================================================================

class World:
    """
    The world consists of a grid of resource tiles and a list of characters.
    Characters move, harvest, reproduce and die.  Each simulation step
    updates both the tiles and the characters.
    """

    def __init__(self, width: int, height: int, initial_characters: int):
        self.width = width
        self.height = height
        # Create resource grid
        self.tiles: List[List[str]] = [[choose_resource() for _ in range(height)] for _ in range(width)]
        # Dictionary for tracking resource regrowth timers: maps (x,y) to (resource, remaining_steps)
        self.regrowth: Dict[Tuple[int, int], Tuple[str, int]] = {}
        # Stockpile locations and counts.  ``stockpile_locations`` maps resource types to coordinates;
        # ``stockpile_counts`` maps coordinates to count of stored items.
        self.stockpile_locations: Dict[str, Tuple[int, int]] = {}
        self.stockpile_counts: Dict[Tuple[int, int], int] = {}
        # Place characters randomly on empty tiles
        self.characters: List[Character] = []
        for _ in range(initial_characters):
            name = generate_name()
            while True:
                x = random.randint(0, width - 1)
                y = random.randint(0, height - 1)
                if self.is_empty(x, y):
                    self.add_character(name, x, y)
                    break

        # After placing characters, create stockpile tiles for each resource type
        self.create_stockpiles()

    def is_empty(self, x: int, y: int) -> bool:
        """Return True if the tile is empty (no character on it)."""
        for c in self.characters:
            if c.x == x and c.y == y:
                return False
        return True

    def add_character(self, name: str, x: int, y: int) -> None:
        """Create a new character at (x,y) and register it."""
        char = Character(name, x, y)
        char.load_state()  # load previous memory if exists
        self.characters.append(char)

    def remove_character(self, char: Character) -> None:
        """Remove a character from the world."""
        if char in self.characters:
            self.characters.remove(char)

    # ------------------------------------------------------------------
    # Stockpile and regrowth management
    # ------------------------------------------------------------------
    def create_stockpiles(self) -> None:
        """
        Place stockpile tiles for each resource defined in ``STOCKPILE_RESOURCES``.

        Each stockpile tile occupies an empty location and tracks the number of
        deposited items.  ``self.stockpile_locations`` maps the resource type
        to its (x, y) coordinate.  ``self.stockpile_counts`` holds the count
        stored at each stockpile coordinate.
        """
        for resource in STOCKPILE_RESOURCES:
            # Find a suitable empty tile
            for _ in range(1000):
                x = random.randint(0, self.width - 1)
                y = random.randint(0, self.height - 1)
                if self.is_empty(x, y) and self.tiles[x][y] != "stockpile":
                    # Designate as stockpile
                    self.tiles[x][y] = "stockpile"
                    self.stockpile_locations[resource] = (x, y)
                    self.stockpile_counts[(x, y)] = 0
                    break

    def update_regrowth(self) -> None:
        """
        Decrement regrowth timers and restore resources when timers expire.

        Resources scheduled for regrowth are stored in ``self.regrowth``.  Each
        entry maps a coordinate to a tuple (resource, steps_remaining).
        When the steps reach zero, the resource is restored to the tile.
        """
        new_regrowth: Dict[Tuple[int, int], Tuple[str, int]] = {}
        for pos, (resource, steps) in self.regrowth.items():
            steps -= 1
            if steps <= 0:
                x, y = pos
                # Only restore if the tile is currently empty; if occupied by
                # another resource, skip regeneration (e.g. if a stockpile was
                # placed here later).
                if self.tiles[x][y] == "empty":
                    self.tiles[x][y] = resource
            else:
                new_regrowth[pos] = (resource, steps)
        self.regrowth = new_regrowth

    def find_nearest(self, char: Character, target_resource: str | List[str]) -> Optional[Tuple[int, int]]:
        """
        Find the nearest tile containing the target resource(s) using BFS.

        ``target_resource`` may be a string or a list of resource strings.  The
        search will return the first tile (other than the character's current
        position) whose resource type matches any element in the list.
        Returns (x, y) of the resource or ``None`` if not found.
        """
        from collections import deque

        # Normalise to a set for efficient membership test
        if isinstance(target_resource, str):
            target_set = {target_resource}
        else:
            target_set = set(target_resource)
        visited = set()
        queue = deque([(char.x, char.y)])
        while queue:
            x, y = queue.popleft()
            if (x, y) in visited:
                continue
            visited.add((x, y))
            # Skip current position for matching
            if (x != char.x or y != char.y) and self.tiles[x][y] in target_set:
                return (x, y)
            # Expand neighbours (N, S, E, W)
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    queue.append((nx, ny))
        return None

    def move_character(self, char: Character, dest: Tuple[int, int]) -> None:
        """Move character one step towards dest (x,y)."""
        dest_x, dest_y = dest
        dx = dy = 0
        if dest_x > char.x:
            dx = 1
        elif dest_x < char.x:
            dx = -1
        if dest_y > char.y:
            dy = 1
        elif dest_y < char.y:
            dy = -1
        # Attempt move in x and y; check emptiness separately
        target_positions = []
        if dx != 0:
            target_positions.append((char.x + dx, char.y))
        if dy != 0:
            target_positions.append((char.x, char.y + dy))
        random.shuffle(target_positions)
        for tx, ty in target_positions:
            if 0 <= tx < self.width and 0 <= ty < self.height and self.is_empty(tx, ty):
                char.x, char.y = tx, ty
                return
        # If no valid move, stay
        return

    def find_nearest_partner(self, char: Character) -> Optional[Tuple[int, int]]:
        """
        Determine the target coordinates of a partner for reproduction.

        If the character has reproduced before, prefer the partner with the highest
        count in the ``mates`` dictionary, provided that partner is still alive.
        Otherwise, find the nearest horny character (by Manhattan distance)
        who meets the reproductive age and satiation requirements.

        Returns the coordinates of the target partner, or ``None`` if none found.
        """
        # Prioritise existing mates
        # Sort mates by reproduction count descending
        for partner_id, _count in sorted(char.mates.items(), key=lambda item: item[1], reverse=True):
            for candidate in self.characters:
                if candidate.id == partner_id and candidate is not char:
                    # Partner may not currently be horny, but we still move toward them
                    return (candidate.x, candidate.y)
        # Otherwise, pick nearest horny partner
        best_dist: Optional[int] = None
        best_pos: Optional[Tuple[int, int]] = None
        for candidate in self.characters:
            if candidate is char:
                continue
            # Candidate must be of reproduction age and currently horny and satiated
            if candidate.age >= REPRODUCTION_AGE and candidate.horny and \
               candidate.hunger > HUNGER_THRESHOLD and candidate.thirst > THIRST_THRESHOLD:
                dist = abs(candidate.x - char.x) + abs(candidate.y - char.y)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_pos = (candidate.x, candidate.y)
        return best_pos
    
    def has_neighbors(self, char: Character) -> bool:
        """Check if the character has any adjacent characters (N, S, E, W)."""
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = char.x + dx, char.y + dy
            if any(c.x == nx and c.y == ny for c in self.characters if c != char):
                return True
        return False

    def character_step(self, char: Character) -> None:
        """
        Update a single character: hunger, thirst, age, movement, harvesting,
        drinking, depositing, reproduction and memory/state persistence.

        Characters age each step and both hunger and thirst decrease.  Their
        behaviour is determined by their current needs and whether they are
        carrying a resource to deposit.  When thirsty, they seek water; when
        hungry, they seek edible resources (trees or plants).  When satiated
        and not carrying anything, they choose a random resource type from
        ``STOCKPILE_RESOURCES`` to collect for stockpiling.  After harvesting
        a resource, they carry it to the appropriate stockpile tile and
        deposit it.
        """
        # Increase age and decrease hunger/thirst
        char.age += 1
        char.hunger -= 1
        char.thirst -= 1
        harvested = False
        drank = False
        deposited = False

        # Update horny state based on satiation and carrying.  A character is horny
        # only when fully fed, fully hydrated and not carrying a resource.
        if char.hunger > HUNGER_THRESHOLD and char.thirst > THIRST_THRESHOLD and char.carrying is None:
            char.horny = True
        else:
            char.horny = False

        # If carrying a resource, head to its stockpile
        if char.carrying:
            dest = self.stockpile_locations.get(char.carrying)
            if dest:
                if (char.x, char.y) == dest:
                    # Deposit the carried resource into the stockpile
                    self.stockpile_counts[dest] = self.stockpile_counts.get(dest, 0) + 1
                    deposited = True
                    char.carrying = None
                    # Clear target so a new one will be chosen when satiated
                    char.stockpile_target = None
                else:
                    # Move one step toward stockpile
                    self.move_character(char, dest)
            # If no dest (shouldn't happen), drop resource and reset
        else:
            # If the character is horny and of reproductive age, attempt to move toward a partner
            partner_dest: Optional[Tuple[int, int]] = None
            if char.horny and char.age >= REPRODUCTION_AGE:
                partner_dest = self.find_nearest_partner(char)
            if partner_dest:
                # Move one step toward the partner
                self.move_character(char, partner_dest)
            else:
                # Determine desire based on thirst and hunger when not seeking a partner
                if char.thirst < THIRST_THRESHOLD:
                    # Seek water when thirsty
                    desire_resources: List[str] = ["water"]
                elif char.hunger < HUNGER_THRESHOLD:
                    # Seek food when hungry (trees or plants)
                    desire_resources = ["tree", "plant"]
                else:
                    # Satiated: choose a resource to stockpile if not already chosen
                    if char.stockpile_target is None:
                        char.stockpile_target = random.choice(STOCKPILE_RESOURCES)
                    desire_resources = [char.stockpile_target]
                dest = self.find_nearest(char, desire_resources)
                if dest:
                    self.move_character(char, dest)
                else:
                    # Random wander if no target found
                    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
                    random.shuffle(dirs)
                    for dx, dy in dirs:
                        nx, ny = char.x + dx, char.y + dy
                        if 0 <= nx < self.width and 0 <= ny < self.height and self.is_empty(nx, ny):
                            char.x, char.y = nx, ny
                            break
            # Evaluate the tile after movement if not carrying
            tile_resource = self.tiles[char.x][char.y]
            # Drinking water when thirsty
            if tile_resource == "water" and char.thirst < THIRST_THRESHOLD:
                char.thirst = random.randint(5, 20)
                # Remove water and schedule regrowth
                self.tiles[char.x][char.y] = "empty"
                self.regrowth[(char.x, char.y)] = ("water", REGROW_STEPS["water"])
                drank = True
            # Eating trees or plants when hungry
            elif tile_resource in ("tree", "plant") and char.hunger < HUNGER_THRESHOLD:
                char.hunger = random.randint(5, 20)
                # Remove resource and schedule regrowth
                self.tiles[char.x][char.y] = "empty"
                self.regrowth[(char.x, char.y)] = (tile_resource, REGROW_STEPS[tile_resource])
                harvested = True
            # Harvesting trees, plants or stone for stockpile when satiated and not seeking partner
            elif tile_resource in ("tree", "plant", "stone"):
                # Only harvest if the resource matches the stockpile target
                # or if no target (for stone we always harvest when satiated)
                if (char.stockpile_target is None) or (tile_resource == char.stockpile_target) or (tile_resource == "stone"):
                    char.carrying = tile_resource
                    # Remove resource and schedule regrowth
                    self.tiles[char.x][char.y] = "empty"
                    self.regrowth[(char.x, char.y)] = (tile_resource, REGROW_STEPS[tile_resource])
                    harvested = True

            # Attempt reproduction after movement and evaluation.  Only if still horny,
            # at reproductive age and satiated, and adjacent to a suitable partner.
            if char.horny and char.age >= REPRODUCTION_AGE and char.hunger > HUNGER_THRESHOLD and char.thirst > THIRST_THRESHOLD:
                # Check adjacent tiles for partner
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = char.x + dx, char.y + dy
                    # Look for partner character at this location
                    for partner in self.characters:
                        if partner is char:
                            continue
                        if partner.x == nx and partner.y == ny and partner.age >= REPRODUCTION_AGE and partner.horny and partner.hunger > HUNGER_THRESHOLD and partner.thirst > THIRST_THRESHOLD and partner.carrying is None:
                            # Reproduction occurs with a probability
                            if random.random() < REPRODUCTION_PROBABILITY:
                                # Find an empty adjacent tile for offspring near either parent
                                placed = False
                                for parent in (char, partner):
                                    for dx2, dy2 in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                                        bx, by = parent.x + dx2, parent.y + dy2
                                        if 0 <= bx < self.width and 0 <= by < self.height and self.is_empty(bx, by):
                                            offspring_name = generate_name()
                                            self.add_character(offspring_name, bx, by)
                                            placed = True
                                            break
                                    if placed:
                                        break
                                # Record mating counts
                                char.mates[partner.id] = char.mates.get(partner.id, 0) + 1
                                partner.mates[char.id] = partner.mates.get(char.id, 0) + 1
                                # Reset horny state for both
                                char.horny = False
                                partner.horny = False
                            # Once interaction attempted, break inner loops
                            break
                    else:
                        # continue outer loop if no partner found at this offset
                        continue
                    # If partner found and reproduction attempted, break the outer loop
                    break
        # Append memory snapshot for this step
        char.memory.append({
            "step": char.age,
            "x": char.x,
            "y": char.y,
            "hunger": char.hunger,
            "thirst": char.thirst,
            "carrying": char.carrying,
            "harvested": harvested,
            "drank": drank,
            "deposited": deposited,
        })
        # Trim memory to appropriate length
        char.update_memory()
        # Persist state to disk
        char.save_state()
        # Death
        if char.age >= DEATH_AGE:
            self.remove_character(char)

    def step(self) -> None:
        """Advance the world by one simulation step."""
        # Make a copy of characters since reproduction may modify the list
        for char in list(self.characters):
            self.character_step(char)
        # Update regrowth timers and restore resources as needed
        self.update_regrowth()


# =============================================================================
# AI log generation
# =============================================================================

def summarize_memory(char: Character) -> str:
    """
    Simulate an AI log generation by summarizing the character's memory.

    This function is a placeholder for an actual AI API call.  It takes
    the character's memory (list of state snapshots) and returns a
    human‑readable string summarising recent events.  The summary
    includes age, hunger status, number of harvests and travels.
    """
    if not char.memory:
        return f"{char.name} has just begun its journey."
    # Analyse memory to summarise hunger, thirst, harvests, drinks and deposits
    total_steps = char.age
    hunger = char.hunger
    thirst = char.thirst
    harvests = sum(1 for s in char.memory if s.get("harvested"))
    drinks = sum(1 for s in char.memory if s.get("drank"))
    deposits = sum(1 for s in char.memory if s.get("deposited"))
    unique_positions = {(s["x"], s["y"]) for s in char.memory}
    log = (
        f"At age {total_steps}, {char.name} feels "
        f"{'thirsty' if thirst < THIRST_THRESHOLD else 'quenched'} and "
        f"{'hungry' if hunger < HUNGER_THRESHOLD else 'satisfied'}. "
        f"It has harvested {harvests} resources, drunk water {drinks} times, deposited {deposits} items "
        f"and visited {len(unique_positions)} unique tiles."
    )
    return log


def generate_ai_log_for_character(char: Character) -> None:
    """
    Append a new AI log entry to the character's JSON file and memory.  This
    function simulates the process of sending the character's memory to an
    AI service and receiving a narrative description.
    """
    summary = summarize_memory(char)
    # Append to in‑memory list
    char.ai_logs.append(summary)
    # Persist to disk
    char.save_state()


# =============================================================================
# GUI
# =============================================================================

class WorldGUI:
    """
    Tkinter interface for displaying and interacting with the world.
    This implementation supports large worlds by rendering only a
    viewport.  The cell size and visible tile count are determined
    dynamically based on the window size.  Users can pan the view with
    arrow keys and zoom in/out with +/- keys.  Characters are drawn on
    top of resource tiles, and a side panel lists all characters.
    """

    def __init__(self, root: tk.Tk):
        self.root = root
        # Create the world
        self.world = World(WORLD_WIDTH, WORLD_HEIGHT, INITIAL_CHARACTER_COUNT)
        self.paused: bool = False  # Simulation pause flag

        # Camera / viewport state
        self.cam_x: int = 0  # top-left tile x
        self.cam_y: int = 0  # top-left tile y
        # Target view dimensions (approximate number of tiles to show); adjust for default zoom
        self.target_view_cols: int = 120  # desired number of columns
        self.target_view_rows: int = 60   # desired number of rows
        # Reserve space for UI
        self.control_bar_h: int = 40
        self.sidebar_w: int = 200
        self.canvas_margin: int = 8

        # Create a top frame for controls
        control_frame = tk.Frame(root)
        control_frame.pack(fill="x")
        # Pause/Resume button
        self.pause_button = tk.Button(control_frame, text="Pause", command=self.toggle_pause)
        self.pause_button.pack(side="left", padx=4, pady=4)

        # Main frame holds the canvas and the info panel
        main_frame = tk.Frame(root)
        main_frame.pack(fill="both", expand=True)

        # Placeholder canvas; size will be set in on_resize
        self.canvas = tk.Canvas(main_frame, highlightthickness=0)
        self.canvas.pack(side="left")

        # Info panel to the right of the canvas
        self.info_panel = tk.Frame(main_frame, width=self.sidebar_w)
        self.info_panel.pack(side="right", fill="y")
        title = tk.Label(self.info_panel, text="Characters", font=("Arial", 12, "bold"))
        title.pack(pady=4)
        # Listbox to display character names
        self.listbox = tk.Listbox(self.info_panel, width=20, activestyle="none")
        self.listbox.pack(fill="both", expand=True, padx=4, pady=4)
        # Bind motion over listbox to highlight characters
        self.listbox.bind("<Motion>", self.on_listbox_motion)
        self.listbox.bind("<Leave>", lambda e: self.clear_highlight())
        # Keep a mapping of listbox indices to characters
        self.listbox_chars: List[Character] = []
        # Variables for highlighting on canvas
        self.highlight_rect_id: Optional[int] = None
        self.highlight_char_id: Optional[int] = None

        # Canvas drawing state
        self.cell_size: int = 10  # will be computed on first resize
        self.view_cols: int = 0
        self.view_rows: int = 0

        # Set up event bindings
        self.canvas.bind("<Button-1>", self.on_click)
        # Pan the camera with arrow keys (or WASD) - coarse movement (10% of viewport)
        self.root.bind("<Left>", lambda e: self.pan(-max(1, self.view_cols // 10), 0))
        self.root.bind("<Right>", lambda e: self.pan(max(1, self.view_cols // 10), 0))
        self.root.bind("<Up>", lambda e: self.pan(0, -max(1, self.view_rows // 10)))
        self.root.bind("<Down>", lambda e: self.pan(0, max(1, self.view_rows // 10)))
        # Zoom with +/-
        self.root.bind("+", lambda e: self.zoom(1.25))
        self.root.bind("=", lambda e: self.zoom(1.25))  # for + without shift
        self.root.bind("-", lambda e: self.zoom(0.8))

        # Redraw on window resize
        self.root.bind("<Configure>", self.on_resize)

        # Track text overlays for characters
        self.char_texts: Dict[int, int] = {}
        # Track text overlays for stockpile counts; maps (x,y) to canvas text id
        self.stockpile_texts: Dict[Tuple[int, int], int] = {}

        # Start simulation loop
        self.update_loop()

        # Perform an initial resize to compute cell size and view dimensions
        self.on_resize()

    # ------------------------------------------------------------------
    # Viewport and scaling helpers
    # ------------------------------------------------------------------
    def compute_cell_size(self) -> int:
        """Compute tile size based on current window size and desired view tiles."""
        # Window dimensions (fallback to screen if window isn't realised yet)
        self.root.update_idletasks()
        win_w = self.root.winfo_width() or self.root.winfo_screenwidth()
        win_h = self.root.winfo_height() or self.root.winfo_screenheight()
        # Available area for canvas
        avail_w = max(200, win_w - self.sidebar_w - 2 * self.canvas_margin)
        avail_h = max(200, win_h - self.control_bar_h - 2 * self.canvas_margin)
        # Compute tile size that fits target tiles
        size_w = avail_w // self.target_view_cols
        size_h = avail_h // self.target_view_rows
        size = max(2, min(size_w, size_h))
        # Limit maximum size for performance
        return min(size, 16)

    def compute_view_tile_dims(self, cell_size: int) -> Tuple[int, int]:
        """Determine how many tiles fit horizontally and vertically."""
        self.root.update_idletasks()
        win_w = self.root.winfo_width() or self.root.winfo_screenwidth()
        win_h = self.root.winfo_height() or self.root.winfo_screenheight()
        avail_w = max(200, win_w - self.sidebar_w - 2 * self.canvas_margin)
        avail_h = max(200, win_h - self.control_bar_h - 2 * self.canvas_margin)
        cols = max(1, min(self.world.width, avail_w // cell_size))
        rows = max(1, min(self.world.height, avail_h // cell_size))
        return cols, rows

    def on_resize(self, _event=None) -> None:
        """Handle window resize: recompute cell size, view dims and redraw."""
        new_size = self.compute_cell_size()
        if new_size != self.cell_size:
            self.cell_size = new_size
            self.view_cols, self.view_rows = self.compute_view_tile_dims(self.cell_size)
            # Resize canvas
            canvas_w = self.view_cols * self.cell_size
            canvas_h = self.view_rows * self.cell_size
            self.canvas.config(width=canvas_w, height=canvas_h)
            # Adjust camera if out of bounds
            self.cam_x = min(self.cam_x, max(0, self.world.width - self.view_cols))
            self.cam_y = min(self.cam_y, max(0, self.world.height - self.view_rows))
            # Redraw world
            self.draw_world()
        else:
            # Still update view dims and canvas in case window grown but cell size unchanged
            new_cols, new_rows = self.compute_view_tile_dims(self.cell_size)
            if new_cols != self.view_cols or new_rows != self.view_rows:
                self.view_cols, self.view_rows = new_cols, new_rows
                self.cam_x = min(self.cam_x, max(0, self.world.width - self.view_cols))
                self.cam_y = min(self.cam_y, max(0, self.world.height - self.view_rows))
                canvas_w = self.view_cols * self.cell_size
                canvas_h = self.view_rows * self.cell_size
                self.canvas.config(width=canvas_w, height=canvas_h)
                self.draw_world()

    def pan(self, dx: int, dy: int) -> None:
        """Move the camera by a given number of tiles."""
        new_x = max(0, min(self.cam_x + dx, self.world.width - self.view_cols))
        new_y = max(0, min(self.cam_y + dy, self.world.height - self.view_rows))
        if new_x != self.cam_x or new_y != self.cam_y:
            self.cam_x, self.cam_y = new_x, new_y
            self.draw_world()

    def zoom(self, factor: float) -> None:
        """Zoom the viewport in (>1) or out (<1) by adjusting target tiles."""
        # Adjust target tile counts
        new_cols = int(self.target_view_cols / factor)
        new_rows = int(self.target_view_rows / factor)
        # Clamp
        self.target_view_cols = max(20, min(300, new_cols))
        self.target_view_rows = max(10, min(200, new_rows))
        # Recompute cell size and dims
        self.on_resize()

    def center_on(self, wx: int, wy: int) -> None:
        """Center the viewport around a given world coordinate."""
        new_cam_x = wx - self.view_cols // 2
        new_cam_y = wy - self.view_rows // 2
        self.cam_x = max(0, min(new_cam_x, self.world.width - self.view_cols))
        self.cam_y = max(0, min(new_cam_y, self.world.height - self.view_rows))
        self.draw_world()

    def toggle_pause(self) -> None:
        """Toggle the paused state of the simulation."""
        self.paused = not self.paused
        # Update button text accordingly
        self.pause_button.configure(text="Resume" if self.paused else "Pause")

    def draw_world(self) -> None:
        """Render the visible portion of the world onto the canvas."""
        cs = self.cell_size
        x0, y0 = self.cam_x, self.cam_y
        x1 = min(self.world.width, x0 + self.view_cols)
        y1 = min(self.world.height, y0 + self.view_rows)

        # Clear canvas
        self.canvas.delete("all")
        # Draw resource tiles within view and prepare stockpile overlay data
        # Clear previous stockpile overlays
        for text_id in list(self.stockpile_texts.values()):
            try:
                self.canvas.delete(text_id)
            except Exception:
                pass
        self.stockpile_texts.clear()

        for wx in range(x0, x1):
            for wy in range(y0, y1):
                px = (wx - x0) * cs
                py = (wy - y0) * cs
                tile = self.world.tiles[wx][wy]
                colour = TILE_COLOURS.get(tile, TILE_COLOURS["empty"])
                self.canvas.create_rectangle(px, py, px + cs, py + cs, fill=colour, outline="")
                # Draw stockpile count overlay if this tile is a stockpile
                if tile == "stockpile":
                    # Only draw if no character occupies this tile
                    occupied = any((c.x == wx and c.y == wy) for c in self.world.characters)
                    if not occupied:
                        count = self.world.stockpile_counts.get((wx, wy), 0)
                        if count > 0 and cs >= 6:
                            text_id = self.canvas.create_text(
                                px + cs // 2, py + cs // 2,
                                text=str(count), fill="black", font=("Arial", max(6, cs - 4))
                            )
                            self.stockpile_texts[(wx, wy)] = text_id

        # Draw characters in view
        self.char_texts.clear()
        for char in self.world.characters:
            if x0 <= char.x < x1 and y0 <= char.y < y1:
                px = (char.x - x0) * cs
                py = (char.y - y0) * cs
                # Draw character overlay rectangle
                self.canvas.create_rectangle(px + 1, py + 1, px + cs - 1, py + cs - 1, fill=TILE_COLOURS["character"], outline="black")
                # Draw first letter for identification if space allows
                if cs >= 6:
                    tid = self.canvas.create_text(px + cs // 2, py + cs // 2, text=char.name[0], fill="white", font=("Arial", max(6, cs - 2)))
                    self.char_texts[char.id] = tid

        # Draw highlight box if applicable
        if self.highlight_char_id is not None:
            # Remove any existing highlight rectangle
            if self.highlight_rect_id is not None:
                try:
                    self.canvas.delete(self.highlight_rect_id)
                except Exception:
                    pass
                self.highlight_rect_id = None
            # Locate the highlighted character
            highlighted_char: Optional[Character] = None
            for c in self.world.characters:
                if c.id == self.highlight_char_id:
                    highlighted_char = c
                    break
            # Draw highlight if the character is within view
            if highlighted_char is not None and x0 <= highlighted_char.x < x1 and y0 <= highlighted_char.y < y1:
                px = (highlighted_char.x - x0) * cs
                py = (highlighted_char.y - y0) * cs
                self.highlight_rect_id = self.canvas.create_rectangle(px, py, px + cs, py + cs, outline="red", width=2)
            else:
                # Character no longer exists or is outside view
                self.clear_highlight()

    def update_loop(self) -> None:
        """Advance the simulation and redraw the world periodically."""
        # Advance world if not paused
        if not self.paused:
            self.world.step()
        # Draw world view
        self.draw_world()
        # Update character list
        self.update_listbox()
        # Schedule next update
        self.root.after(UPDATE_INTERVAL_MS, self.update_loop)

    def on_click(self, event) -> None:
        """Handle canvas clicks; map view coords to world coords and show logs."""
        cs = self.cell_size
        wx = self.cam_x + event.x // cs
        wy = self.cam_y + event.y // cs
        if 0 <= wx < self.world.width and 0 <= wy < self.world.height:
            for char in self.world.characters:
                if char.x == wx and char.y == wy:
                    self.show_character_popup(char)
                    return

    # ----------------------------------------------------------------------
    # Side panel methods
    # ----------------------------------------------------------------------
    def update_listbox(self) -> None:
        """Refresh the listbox contents to reflect current characters."""
        # Preserve currently highlighted character id
        current_highlight_id = self.highlight_char_id
        self.listbox.delete(0, tk.END)
        self.listbox_chars = list(self.world.characters)
        for idx, char in enumerate(self.listbox_chars):
            self.listbox.insert(idx, f"{char.name} (ID {char.id})")
        # Restore highlight if applicable
        if current_highlight_id is not None:
            for idx, char in enumerate(self.listbox_chars):
                if char.id == current_highlight_id:
                    # Optionally select the item
                    # self.listbox.selection_clear(0, tk.END)
                    # self.listbox.selection_set(idx)
                    break

    def on_listbox_motion(self, event) -> None:
        """Handle mouse movement over the listbox to highlight corresponding character."""
        # Determine index under cursor
        idx = self.listbox.nearest(event.y)
        # Only proceed if index is valid
        if idx < 0 or idx >= len(self.listbox_chars):
            self.clear_highlight()
            return
        char = self.listbox_chars[idx]
        if char.id != self.highlight_char_id:
            self.highlight_character(char)

    def highlight_character(self, char: Character) -> None:
        """Highlight the given character's tile on the canvas relative to the viewport."""
        # Set the highlighted character and recenter view if needed
        self.highlight_char_id = char.id
        # Center on character; this also triggers draw_world
        self.center_on(char.x, char.y)

    def clear_highlight(self) -> None:
        """Remove any highlight on the canvas and reset state."""
        if self.highlight_rect_id is not None:
            try:
                self.canvas.delete(self.highlight_rect_id)
            except Exception:
                pass
            self.highlight_rect_id = None
        self.highlight_char_id = None

    def show_character_popup(self, char: Character) -> None:
        """
        Display a small popup showing the full JSON state of the character.

        When the user clicks on a character, the simulation pauses and a
        window pops up containing a JSON dump of the character's current
        state.  The JSON includes position, hunger, thirst, carrying
        status, mates information and memory.  No AI log generation is
        performed here; this function simply prints the raw state so the
        player can understand what the character is doing.
        """
        popup = tk.Toplevel(self.root)
        popup.title(f"{char.name} (ID {char.id}) State")
        popup.geometry("400x300")
        popup.transient(self.root)
        popup.grab_set()

        # Pause the simulation if not already paused
        if not self.paused:
            self.toggle_pause()

        # Ensure the character's state is saved to disk (memory, logs etc.)
        # Save current state for up-to-date information
        char.save_state()

        # Build a dictionary of the character's current state for display
        state_data = {
            "id": char.id,
            "name": char.name,
            "age": char.age,
            "position": {"x": char.x, "y": char.y},
            "hunger": char.hunger,
            "thirst": char.thirst,
            "carrying": char.carrying,
            "stockpile_target": char.stockpile_target,
            "horny": char.horny,
            "mates": char.mates,
            "memory": char.memory,
        }
        # Convert to formatted JSON string
        json_text = json.dumps(state_data, indent=2)

        # Display the JSON in a read-only text widget
        text_widget = tk.Text(popup, wrap="word", height=15, width=50)
        text_widget.pack(padx=10, pady=10, fill="both", expand=True)
        text_widget.insert("1.0", json_text)
        text_widget.configure(state="disabled")

        # When popup is closed, resume simulation if paused
        def on_close() -> None:
            # Destroy the popup and resume simulation
            try:
                popup.destroy()
            finally:
                if self.paused:
                    self.toggle_pause()

        popup.protocol("WM_DELETE_WINDOW", on_close)


def main() -> None:
    """Entry point to launch the GUI."""
    root = tk.Tk()
    root.title("Dwarf‑like Pixel World")
    gui = WorldGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()