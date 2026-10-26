from fnmatch import fnmatch
from collections import defaultdict
# Assuming Heuristic base class is available in this path
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class for standalone testing if needed
# In the actual environment, this import will work.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        """Dummy Heuristic base class for standalone testing."""
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
        def __call__(self, node):
            raise NotImplementedError("Subclass must implement abstract method")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure fact is a string and remove leading/trailing parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, maybe raise an error or return None/empty list
        # For robustness, let's return empty list if format is unexpected
        return []
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def get_coords(tile_name):
    """
    Parses a tile name like 'tile_row_col' into integer coordinates (row, col).
    Assumes tile names follow this specific format.
    """
    try:
        parts = tile_name.split('_')
        if len(parts) == 3 and parts[0] == 'tile':
            row = int(parts[1])
            col = int(parts[2])
            return (row, col)
        else:
            # Handle unexpected tile name format
            # print(f"Warning: Unexpected tile name format: {tile_name}")
            return None # Or raise an error
    except (ValueError, IndexError):
        # Handle cases where parts are not integers or list indexing fails
        # print(f"Warning: Could not parse coordinates from tile name: {tile_name}")
        return None


def ManhattanDistance(coords1, coords2):
    """Calculates the Manhattan distance between two coordinate pairs (r, c)."""
    if coords1 is None or coords2 is None:
        return float('inf') # Cannot calculate distance for invalid coordinates
    r1, c1 = coords1
    r2, c2 = coords2
    return abs(r1 - r2) + abs(c1 - c2)


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    Estimates the cost to reach the goal by summing the minimum estimated cost
    for each tile that needs to be painted correctly. The estimated cost for
    a single tile considers the closest robot, the cost for that robot to
    acquire the correct color, and the movement cost to get adjacent to the tile,
    plus the paint action itself.

    This heuristic is non-admissible as it sums independent minimum costs,
    ignoring resource contention (multiple tiles needing the same robot)
    and potential synergies (a robot painting multiple tiles in one trip).
    It aims to be more informed than a simple goal count by incorporating
    color and location aspects.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and static facts.
        Precomputes which tiles can be painted from which adjacent tiles.
        """
        super().__init__(task)

        # Extract goal painted states: {tile: color, ...}
        self.goal_painted = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'painted':
                if len(parts) == 3:
                    tile, color = parts[1], parts[2]
                    self.goal_painted[tile] = color
                # else: Handle unexpected goal format if necessary

        # Build map: {tile_to_paint: {tile_robot_must_be_at}, ...}
        # This captures the adjacency required for painting.
        self.paintable_from = defaultdict(set)
        for fact in self.static:
            parts = get_parts(fact)
            if parts and parts[0] in ['up', 'down', 'left', 'right']:
                if len(parts) == 3:
                    tile_to_paint = parts[1]
                    tile_robot_must_be_at = parts[2]
                    self.paintable_from[tile_to_paint].add(tile_robot_must_be_at)
                # else: Handle unexpected static fact format if necessary

        # Cache coordinates for all tiles mentioned in paintable_from or goals
        self._tile_coords_cache = {}
        all_relevant_tiles = set(self.goal_painted.keys())
        for tile_set in self.paintable_from.values():
             all_relevant_tiles.update(tile_set)
        for tile in all_relevant_tiles:
            self._tile_coords_cache[tile] = get_coords(tile)


    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        Estimates the total cost based on unpainted goal tiles.
        """
        state = node.state

        # Parse current state
        robot_state = {} # {robot: (location, color), ...}
        tile_is_clear = {} # {tile: bool, ...}
        tile_painted_color = {} # {tile: color, ...}

        # Collect all tiles mentioned in the state to initialize tile_is_clear
        all_tiles_in_state = set()
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'robot-at' and len(parts) == 3:
                robot, location = parts[1], parts[2]
                if robot not in robot_state:
                    robot_state[robot] = [None, None] # [location, color]
                robot_state[robot][0] = location
                all_tiles_in_state.add(location)
            elif predicate == 'robot-has' and len(parts) == 3:
                 robot, color = parts[1], parts[2]
                 if robot not in robot_state:
                    robot_state[robot] = [None, None] # [location, color]
                 robot_state[robot][1] = color
            elif predicate == 'clear' and len(parts) == 2:
                tile = parts[1]
                tile_is_clear[tile] = True
                all_tiles_in_state.add(tile)
            elif predicate == 'painted' and len(parts) == 3:
                tile, color = parts[1], parts[2]
                tile_painted_color[tile] = color
                all_tiles_in_state.add(tile)
            # Add any other predicates that might mention tiles if necessary
            # For this domain, robot-at, clear, painted cover tile locations/states

        # Initialize tiles not explicitly mentioned as clear=False, painted=None
        for tile in all_tiles_in_state:
             if tile not in tile_is_clear and tile not in tile_painted_color:
                 # If a tile is not clear and not painted, something is wrong
                 # or it's implicitly not clear. Assuming clear is explicitly stated.
                 # If a tile is not mentioned as clear or painted, its state is unknown.
                 # Based on domain, tiles are either clear or painted.
                 # Let's assume if not clear, it must be painted (or vice versa).
                 # A safer approach is to assume tiles not in clear or painted facts
                 # are in some default state, but the domain implies they are either clear or painted.
                 # Let's rely on the explicit facts provided in the state.
                 pass # State facts are exhaustive for tile state

        # Identify tiles that need to be painted correctly
        tiles_to_paint = {} # {tile: goal_color, ...}
        large_unsolvable_cost = 1000000 # Use a large number for unsolvable states

        for tile, goal_color in self.goal_painted.items():
            current_painted_color = tile_painted_color.get(tile)
            is_clear = tile_is_clear.get(tile, False) # Default to False if not in clear facts

            if current_painted_color == goal_color:
                # Tile is already painted correctly
                continue
            elif current_painted_color is not None and current_painted_color != goal_color:
                # Tile is painted with the wrong color - likely unsolvable given domain actions
                # The paint action requires the tile to be 'clear'.
                return large_unsolvable_cost
            elif is_clear:
                 # Tile is clear but needs painting
                 tiles_to_paint[tile] = goal_color
            # else: Tile is not clear, not painted with goal color, and not painted with wrong color?
            # This case shouldn't happen in valid states based on domain predicates.
            # If a tile is not clear, it must be painted.

        # If all goal tiles are painted correctly, heuristic is 0
        if not tiles_to_paint:
            return 0

        # Calculate heuristic based on tiles that need painting
        total_heuristic = 0

        for tile_to_paint, needed_color in tiles_to_paint.items():
            min_cost_for_this_tile = float('inf')

            # Get coordinates of the tile to be painted
            tile_coords = self._tile_coords_cache.get(tile_to_paint)
            if tile_coords is None:
                 # Should not happen if __init__ is correct, but handle defensively
                 # print(f"Error: Could not get coordinates for tile {tile_to_paint}")
                 return large_unsolvable_cost # Indicate problem

            # Find tiles adjacent to tile_to_paint that a robot must be at
            possible_robot_locations = self.paintable_from.get(tile_to_paint, set())
            if not possible_robot_locations:
                 # This tile cannot be painted according to static facts - likely unsolvable
                 # print(f"Error: No paintable_from locations for tile {tile_to_paint}")
                 return large_unsolvable_cost

            # Find the minimum cost for any robot to paint this tile
            for robot, (robot_location, robot_color) in robot_state.items():
                if robot_location is None or robot_color is None:
                    # Robot state is incomplete, skip or handle error
                    continue

                # Cost to get the correct color
                color_cost = 1 if robot_color != needed_color else 0

                # Minimum movement cost to get to any adjacent tile
                min_move_cost_to_adjacent = float('inf')
                robot_coords = self._tile_coords_cache.get(robot_location)

                if robot_coords is not None:
                    for adj_tile in possible_robot_locations:
                        adj_coords = self._tile_coords_cache.get(adj_tile)
                        if adj_coords is not None:
                            move_cost = ManhattanDistance(robot_coords, adj_coords)
                            min_move_cost_to_adjacent = min(min_move_cost_to_adjacent, move_cost)

                # If no valid adjacent tile or coordinates found, this robot can't paint it
                if min_move_cost_to_adjacent == float('inf'):
                    continue # Try next robot

                # Total estimated cost for this robot to paint this tile
                # color_cost + move_cost + paint_action_cost
                robot_total_cost = color_cost + min_move_cost_to_adjacent + 1

                min_cost_for_this_tile = min(min_cost_for_this_tile, robot_total_cost)

            # If no robot can paint this tile (e.g., no robots, or no path to adjacent), unsolvable
            if min_cost_for_this_tile == float('inf'):
                 # print(f"Error: No robot can paint tile {tile_to_paint}")
                 return large_unsolvable_cost

            total_heuristic += min_cost_for_this_tile

        return total_heuristic

# Example usage (requires a dummy Task and Node class)
# if __name__ == '__main__':
#     class DummyTask:
#         def __init__(self, name, facts, initial_state, goals, operators, static):
#             self.name = name
#             self.facts = facts
#             self.initial_state = frozenset(initial_state)
#             self.goals = frozenset(goals)
#             self.operators = operators
#             self.static = frozenset(static)

#     class DummyNode:
#         def __init__(self, state):
#             self.state = frozenset(state)

#     # Example 1 from problem description
#     task1_static = [
#         '(available-color white)', '(available-color black)',
#         '(up tile_1_1 tile_0_1 )', '(down tile_0_1 tile_1_1 )'
#     ]
#     task1_init = [
#         '(robot-at robot1 tile_0_1)', '(robot-has robot1 black)',
#         '(clear tile_1_1)'
#     ]
#     task1_goals = [
#         '(painted tile_1_1 white)'
#     ]
#     task1 = DummyTask("floortile-01", [], task1_init, task1_goals, [], task1_static)

#     # Test initial state
#     node1_init = DummyNode(task1.initial_state)
#     h1 = floortileHeuristic(task1)
#     print(f"Heuristic for task1 initial state: {h1(node1_init)}") # Expected: > 0

#     # Test a state closer to goal (robot has white, is at tile_0_1)
#     task1_state_closer = [
#         '(robot-at robot1 tile_0_1)', '(robot-has robot1 white)',
#         '(clear tile_1_1)',
#         '(available-color white)', '(available-color black)',
#         '(up tile_1_1 tile_0_1 )', '(down tile_0_1 tile_1_1 )'
#     ]
#     node1_closer = DummyNode(task1_state_closer)
#     print(f"Heuristic for task1 closer state: {h1(node1_closer)}") # Expected: lower than init state

#     # Test goal state
#     task1_goal_state = [
#         '(robot-at robot1 tile_0_1)', # Robot location doesn't matter in goal
#         '(robot-has robot1 white)', # Robot color doesn't matter in goal
#         '(painted tile_1_1 white)',
#         # Note: clear tile_1_1 is false in goal state, but not explicitly stated in PDDL goal
#         # The heuristic only checks for the positive goal predicates.
#         # A full state check would ensure (clear tile_1_1) is false.
#         # For heuristic, we only care if (painted tile_1_1 white) is true.
#         '(available-color white)', '(available-color black)',
#         '(up tile_1_1 tile_0_1 )', '(down tile_0_1 tile_1_1 )'
#     ]
#     node1_goal = DummyNode(task1_goal_state)
#     print(f"Heuristic for task1 goal state: {h1(node1_goal)}") # Expected: 0

#     # Example 2 (partial test)
#     # Need to construct DummyTask for example 2 to test properly
#     # This requires parsing the full example 2 instance file which is complex.
#     # The logic for parsing state and static facts seems correct based on the structure.
#     pass
