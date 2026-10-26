import re
from heuristics.heuristic_base import Heuristic
from task import Task # Assuming Task class is available via task module
# import logging # Import logging for potential debugging

# Configure logging if needed
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

class floortileHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Floortile domain.

    Summary:
        This heuristic estimates the cost to reach the goal by simulating a
        greedy assignment of unpainted goal tiles to robots. In each step of
        the simulation, it finds the unpainted goal tile and robot pair that
        minimizes the cost to paint that tile (movement cost + color change
        cost + paint action cost). It adds this minimum cost to the total
        heuristic value, removes the tile from the set of unpainted tiles,
        and updates the robot's simulated state (location and color). This
        process repeats until all goal tiles are painted in the simulation.
        The heuristic is non-admissible and designed for greedy best-first search.

    Assumptions:
        - Tile names follow the format 'tile_row_col' where row and col are integers.
        - The grid defined by adjacency predicates is connected.
        - All colors required by the goal are available colors.
        - Tiles painted with the wrong color in the current state indicate a dead end.
        - The heuristic simulation ignores the 'clear' precondition for movement
          and painting, assuming that necessary tiles can be made clear.

    Heuristic Initialization:
        The constructor parses the static facts from the task to build:
        - A mapping from tile names to their (row, col) coordinates.
        - An adjacency map between tile names based on up/down/left/right predicates.
        - A mapping from goal tile names to their required color.
        - A set of available colors.
        It also checks if all required goal colors are available, marking the problem
        as unsolvable if not.

    Step-By-Step Thinking for Computing Heuristic:
        1.  Identify all goal facts of the form `(painted T C)`. Store these as
            required paintings {T: C}.
        2.  Iterate through the current state facts.
        3.  Identify the current location and color of each robot.
        4.  Check for dead ends: If any tile T is painted with color C' in the
            current state, but the goal requires it to be painted with a different
            color C, return infinity.
        5.  Identify the set of goal tiles that are not yet painted correctly
            in the current state.
        6.  If there are no unpainted goal tiles, the heuristic is 0.
        7.  Initialize the total heuristic cost `h = 0`.
        8.  Create a mutable copy of the robot states (location, color) for the simulation.
        9.  Create a mutable list of the unpainted goal tiles (tile, required_color).
        10. While there are still unpainted goal tiles:
            a.  Find the minimum cost to paint any of the remaining unpainted tiles
                by any of the robots in their current simulated states.
            b.  The cost for a robot R at L_R with color C_R to paint tile T with color C is:
                `Manhattan_distance(L_R, L_adj) + (1 if C_R != C else 0) + 1`
                where L_adj is any tile adjacent to T, and we take the minimum
                Manhattan distance over all L_adj adjacent to T.
            c.  Select the robot R* and tile T* (with required color C*) that yield
                this minimum cost.
            d.  Add the minimum cost to `h`.
            e.  Remove (T*, C*) from the list of remaining unpainted tiles.
            f.  Update the simulated state of robot R*: its new location becomes
                the adjacent tile L_adj* that minimized the distance to T*, and
                its new color becomes C*.
            g.  If no robot can paint any remaining tile (e.g., required color not available),
                return infinity.
        11. Return the total heuristic cost `h`.
    """

    def __init__(self, task: Task):
        super().__init__()
        self.task = task
        self.tile_coords = {}
        self.adjacency = {}
        self.goal_tiles = {} # {tile_name: color}
        self.available_colors = set()
        self.is_unsolvable = False # Flag for impossible goals (e.g., color not available)

        # Regex to parse tile names like 'tile_row_col'
        self._tile_name_pattern = re.compile(r'tile_(\d+)_(\d+)')

        # Build tile coordinates and adjacency map from static facts
        all_tile_names = set()
        for fact in task.static:
            fact_str = str(fact)
            # Parse adjacency facts
            match_adj = re.match(r'\((up|down|left|right) (\S+) (\S+)\)', fact_str)
            if match_adj:
                t1 = match_adj.group(2)
                t2 = match_adj.group(3)

                all_tile_names.add(t1)
                all_tile_names.add(t2)

                # Populate tile_coords
                if t1 not in self.tile_coords:
                    match_t1 = self._tile_name_pattern.match(t1)
                    if match_t1:
                        self.tile_coords[t1] = (int(match_t1.group(1)), int(match_t1.group(2)))
                if t2 not in self.tile_coords:
                    match_t2 = self._tile_name_pattern.match(t2)
                    if match_t2:
                        self.tile_coords[t2] = (int(match_t2.group(1)), int(match_t2.group(2)))

                # Populate adjacency
                if t1 not in self.adjacency:
                    self.adjacency[t1] = set()
                if t2 not in self.adjacency:
                    self.adjacency[t2] = set()
                self.adjacency[t1].add(t2)
                self.adjacency[t2].add(t1)

            # Parse available-color facts
            match_color = re.match(r'\(available-color (\S+)\)', fact_str)
            if match_color:
                color = match_color.group(1)
                self.available_colors.add(color)

        # Populate goal tiles
        for goal_fact in task.goals:
            goal_str = str(goal_fact)
            match_painted = re.match(r'\(painted (\S+) (\S+)\)', goal_str)
            if match_painted:
                tile = match_painted.group(1)
                color = match_painted.group(2)
                self.goal_tiles[tile] = color

        # Check if all goal colors are available
        required_goal_colors = set(self.goal_tiles.values())
        if not required_goal_colors.issubset(self.available_colors):
             # This indicates an unsolvable problem based on available colors
             self.is_unsolvable = True
             # logging.warning(f"Problem unsolvable: Required colors {required_goal_colors - self.available_colors} are not available.")


    def _get_tile_coords(self, tile_name):
        """Helper to get coordinates, handling potential missing tiles gracefully."""
        return self.tile_coords.get(tile_name)

    def manhattan_distance(self, tile1_name, tile2_name):
        """Calculates Manhattan distance between two tiles."""
        coords1 = self._get_tile_coords(tile1_name)
        coords2 = self._get_tile_coords(tile2_name)
        if coords1 is None or coords2 is None:
            # Should not happen if all tiles in adjacency facts have parsable names
            # and are added to self.tile_coords.
            # logging.error(f"Could not get coordinates for {tile1_name} or {tile2_name}")
            return float('inf') # Indicate an issue
        r1, c1 = coords1
        r2, c2 = coords2
        return abs(r1 - r2) + abs(c1 - c2)

    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for the given state node.
        """
        if self.is_unsolvable:
            return float('inf')

        state = node.state

        # 1. Identify robot states and check for dead ends (wrong painted tiles)
        robot_states = {} # {robot_name: [location, color]}
        painted_tiles_in_state = {} # {tile_name: color}

        for fact in state:
            fact_str = str(fact)
            match_robot_at = re.match(r'\(robot-at (\S+) (\S+)\)', fact_str)
            if match_robot_at:
                robot = match_robot_at.group(1)
                location = match_robot_at.group(2)
                if robot not in robot_states:
                    robot_states[robot] = [None, None]
                robot_states[robot][0] = location
            else:
                match_robot_has = re.match(r'\(robot-has (\S+) (\S+)\)', fact_str)
                if match_robot_has:
                    robot = match_robot_has.group(1)
                    color = match_robot_has.group(2)
                    if robot not in robot_states:
                        robot_states[robot] = [None, None]
                    robot_states[robot][1] = color
                else:
                    match_painted = re.match(r'\(painted (\S+) (\S+)\)', fact_str)
                    if match_painted:
                        tile = match_painted.group(1)
                        color = match_painted.group(2)
                        painted_tiles_in_state[tile] = color

        # Check for dead ends: tile painted with wrong color
        for goal_tile, required_color in self.goal_tiles.items():
            if goal_tile in painted_tiles_in_state and painted_tiles_in_state[goal_tile] != required_color:
                # logging.debug(f"Dead end: {goal_tile} painted with wrong color {painted_tiles_in_state[goal_tile]}, needs {required_color}")
                return float('inf') # Tile painted with wrong color

        # 2. Identify unpainted goal tiles
        unpainted_goal_tiles = [] # List of (tile_name, required_color)
        for goal_tile, required_color in self.goal_tiles.items():
            # Check if the tile is painted with the correct color in the current state
            if not (goal_tile in painted_tiles_in_state and painted_tiles_in_state[goal_tile] == required_color):
                 unpainted_goal_tiles.append((goal_tile, required_color))

        # 3. If goal is reached, heuristic is 0
        if not unpainted_goal_tiles:
            return 0

        # 4. Simulate greedy assignment
        h = 0
        # Create a mutable copy of robot states for simulation {robot_name: [location, color]}
        sim_robot_states = {R: list(state) for R, state in robot_states.items()}
        # Create a list of remaining tiles to paint (tile_name, required_color)
        remaining_tiles = list(unpainted_goal_tiles)

        while remaining_tiles:
            min_total_cost = float('inf')
            best_assignment = None # (robot_name, tile_idx_in_remaining, robot_final_adj_location)

            for tile_idx, (T, C) in enumerate(remaining_tiles):
                # Find tiles adjacent to T
                adjacent_to_T = self.adjacency.get(T, set())
                if not adjacent_to_T:
                    # This tile T has no adjacent tiles? Indicates a problem.
                    # logging.error(f"Tile {T} has no adjacent tiles defined.")
                    return float('inf') # Treat as unsolvable

                for R, (L_R, C_R) in sim_robot_states.items():
                    if L_R is None or C_R is None:
                         # Robot state not fully initialized? Should not happen with valid states.
                         # logging.warning(f"Robot {R} state incomplete: location={L_R}, color={C_R}")
                         continue

                    # Cost to get the required color C
                    cost_color = 0
                    if C_R != C:
                        if C in self.available_colors:
                            cost_color = 1
                        else:
                            # Required color is not available, problem is unsolvable
                            # This should ideally be caught in __init__, but double-check here.
                            # logging.error(f"Required color {C} for tile {T} is not available.")
                            return float('inf')

                    # Find minimum distance from robot's current location L_R to any tile adjacent to T
                    min_dist_to_adj = float('inf')
                    target_adj_location = None

                    for L_adj in adjacent_to_T:
                        dist = self.manhattan_distance(L_R, L_adj)
                        if dist < min_dist_to_adj:
                            min_dist_to_adj = dist
                            target_adj_location = L_adj

                    if target_adj_location is None:
                         # Should not happen if adjacent_to_T was not empty and manhattan_distance is finite
                         # logging.error(f"Could not find a reachable adjacent tile for {T} from {L_R}")
                         continue # Skip this robot/tile pair

                    # Total cost for this robot to paint this tile in this step
                    # move_cost + color_change_cost + paint_action_cost
                    current_assignment_cost = min_dist_to_adj + cost_color + 1

                    if current_assignment_cost < min_total_cost:
                        min_total_cost = current_assignment_cost
                        best_assignment = (R, tile_idx, target_adj_location)

            if best_assignment is None:
                 # This means no robot could be assigned to paint any remaining tile.
                 # This could happen if, e.g., a required color is not available (already checked),
                 # or if a tile has no adjacent tiles (checked), or if no robots exist (problem setup).
                 # Treat as unsolvable.
                 # logging.error("No robot could be assigned to paint any remaining tile.")
                 return float('inf')

            # Apply the best assignment in the simulation
            best_robot, best_tile_idx, robot_final_adj_location = best_assignment
            h += min_total_cost

            # Update simulated robot state: new location is the adjacent tile, new color is the painted color
            painted_tile_info = remaining_tiles.pop(best_tile_idx)
            painted_color = painted_tile_info[1]
            sim_robot_states[best_robot] = [robot_final_adj_location, painted_color]

        return h
