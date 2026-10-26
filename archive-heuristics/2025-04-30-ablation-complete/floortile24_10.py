# Assume Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing three components:
    1. The number of tiles that need to be painted correctly according to the goal.
    2. The number of required colors (for the unpainted goal tiles) that no robot currently possesses.
    3. The sum of minimum movement costs for each unpainted goal tile, where the movement cost for a tile is the minimum distance from any robot's current location to any tile adjacent to the target tile.

    # Assumptions
    - The grid defined by up/down/left/right predicates is connected.
    - Action costs are uniform (implicitly 1).

    # Heuristic Initialization
    - Extracts the goal conditions to identify target tiles and their required colors.
    - Builds an adjacency list representation of the tile grid graph based on static up/down/left/right facts. This graph is used for calculating movement distances.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of tiles that are specified in the goal but are not currently painted with the correct color in the state. Let this set be `MismatchedTiles`.
    2. The first component of the heuristic is the number of paint actions required, which is simply the count of tiles in `MismatchedTiles`. Add `|MismatchedTiles|` to the total heuristic.
    3. Identify the set of colors required by the tiles in `MismatchedTiles`.
    4. Check which of these required colors are not currently held by any robot in the state. Let this set be `MissingColors`.
    5. The second component of the heuristic is the number of color changes required, which is the count of colors in `MissingColors`. Add `|MissingColors|` to the total heuristic.
    6. Calculate the movement cost component:
        - Initialize `total_movement_cost = 0`.
        - For each tile `T` in `MismatchedTiles`:
            - Determine the set of tiles adjacent to `T` using the pre-built adjacency list.
            - Find the minimum distance from *any* robot's current location to *any* tile in the set of adjacent tiles of `T`. This requires performing BFS starting from each robot's location to find the shortest path to any of the adjacent tiles.
            - Add this minimum distance to `total_movement_cost`.
    7. Add `total_movement_cost` to the total heuristic.
    8. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building the tile graph.
        """
        # Ensure the base class constructor is called
        super().__init__(task)

        # Store goal locations and colors for each tile.
        self.goal_painted_tiles = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted" and len(args) == 2:
                tile, color = args
                self.goal_painted_tiles[tile] = color

        # Build the adjacency list for the tile grid graph.
        self.adj = {}
        for fact in self.static: # Use self.static from base class
            parts = get_parts(fact)
            # Check for predicates defining adjacency and ensure correct number of arguments
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                # Predicate (dir y x) means y is in direction dir from x.
                # So y and x are adjacent.
                tile_y, tile_x = parts[1], parts[2]
                self.adj.setdefault(tile_x, []).append(tile_y)
                self.adj.setdefault(tile_y, []).append(tile_x)

        # Remove duplicates from adjacency lists
        for tile in self.adj:
            self.adj[tile] = list(set(self.adj[tile]))

    def get_distance(self, start_tile, target_tiles):
        """
        Calculate the minimum distance from start_tile to any tile in target_tiles
        using BFS on the pre-built adjacency list.
        Returns float('inf') if no target tile is reachable.
        """
        if not target_tiles:
             return float('inf') # No targets to reach

        target_set = set(target_tiles)
        # Check if start is already a target
        if start_tile in target_set:
            return 0

        queue = deque([(start_tile, 0)])
        visited = {start_tile}

        while queue:
            current_tile, dist = queue.popleft()

            # Check if current tile is one of the targets
            if current_tile in target_set:
                return dist

            # Explore neighbors
            for neighbor in self.adj.get(current_tile, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # If BFS completes without reaching any target
        return float('inf')

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic value for the given state.
        """
        state = node.state  # Current world state (frozenset of fact strings)

        # Extract current state information
        robot_locations = {}
        robot_colors = {}
        painted_tiles_state = {}
        # clear_tiles_state = set() # Not directly needed for this heuristic logic

        for fact in state:
            parts = get_parts(fact)
            if not parts: # Handle potential empty fact string if any
                continue
            predicate = parts[0]
            if predicate == "robot-at" and len(parts) == 3:
                robot, tile = parts[1], parts[2]
                robot_locations[robot] = tile
            elif predicate == "robot-has" and len(parts) == 3:
                robot, color = parts[1], parts[2]
                robot_colors[robot] = color
            elif predicate == "painted" and len(parts) == 3:
                tile, color = parts[1], parts[2]
                painted_tiles_state[tile] = color
            # elif predicate == "clear" and len(parts) == 2:
            #     tile = parts[1]
            #     clear_tiles_state.add(tile)

        # 1. Identify tiles that need painting
        mismatched_tiles = set()
        required_colors = set()

        for goal_tile, goal_color in self.goal_painted_tiles.items():
            # Check if the tile is painted correctly in the current state
            if goal_tile not in painted_tiles_state or painted_tiles_state[goal_tile] != goal_color:
                 # Tile needs painting (either not painted or painted wrong color)
                 mismatched_tiles.add(goal_tile)
                 required_colors.add(goal_color)

        # If all goal tiles are painted correctly, the heuristic is 0.
        if not mismatched_tiles:
            return 0

        # Initialize heuristic value
        h = 0

        # Component 1: Paint actions needed (one per mismatched tile)
        h += len(mismatched_tiles)

        # Component 2: Color changes needed
        # Count colors required by mismatched tiles that no robot currently has.
        current_robot_colors = set(robot_colors.values())
        missing_colors = required_colors - current_robot_colors
        h += len(missing_colors)

        # Component 3: Movement cost
        # Sum of minimum distances for each mismatched tile:
        # min_distance from any robot to any tile adjacent to the mismatched tile.
        total_movement_cost = 0
        for tile_to_paint in mismatched_tiles:
            # Find tiles adjacent to the one needing paint
            adjacent_tiles = self.adj.get(tile_to_paint, [])

            if not adjacent_tiles:
                 # This tile has no adjacent tiles, cannot be painted. Problem likely unsolvable.
                 return float('inf')

            # Find the minimum distance from any robot to any adjacent tile
            min_dist_to_adj_for_this_tile = float('inf')

            # If there are no robots, this task is impossible
            if not robot_locations:
                 return float('inf')

            for robot, robot_loc in robot_locations.items():
                 # Calculate distance from this robot's location to any adjacent tile
                 dist_from_this_robot = self.get_distance(robot_loc, adjacent_tiles)
                 min_dist_to_adj_for_this_robot = dist_from_this_robot # Renaming for clarity

                 # Update the minimum distance found across all robots for this tile
                 min_dist_to_adj_for_this_tile = min(min_dist_to_adj_for_this_tile, min_dist_to_adj_for_this_robot)

            # If no robot can reach an adjacent tile for this mismatched tile, problem likely unsolvable.
            if min_dist_to_adj_for_this_tile == float('inf'):
                 return float('inf')

            total_movement_cost += min_dist_to_adj_for_this_tile

        h += total_movement_cost

        return h
