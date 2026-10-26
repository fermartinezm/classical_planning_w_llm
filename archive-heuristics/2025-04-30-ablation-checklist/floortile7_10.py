from fnmatch import fnmatch
from collections import defaultdict, deque
import math # Use math.inf for unreachable distances

# Assuming a base class Heuristic exists in a 'heuristics' directory
# from heuristics.heuristic_base import Heuristic

# If the base class is not provided, define a minimal one for the code to be runnable
class Heuristic:
    """Minimal base class structure assumed by the problem description."""
    def __init__(self, task):
        self.task = task
        # Task object is assumed to have .goals and .static attributes
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        """
        Compute the heuristic value for a given state node.
        Node object is assumed to have a .state attribute (frozenset of facts).
        """
        raise NotImplementedError("Heuristic subclass must implement __call__")


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2:
        return []
    # Remove outer parentheses and split by whitespace
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(predicate arg1 arg2)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of arguments in the pattern
    if len(parts) != len(args):
        return False

    # Check if each part matches the corresponding argument pattern
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all goal tiles
    with their target colors. It sums the minimum cost for each unpainted goal tile
    independently, considering the closest robot that can obtain the required color.

    # Assumptions:
    - Tiles form a grid structure defined by up/down/left/right predicates.
    - Robots can only move onto clear tiles.
    - Painting a tile requires the robot to be on a tile adjacent to the target tile,
      and the target tile must be clear.
    - Robots can change color if the target color is available (cost 1).
    - The cost of moving one step is 1 action.
    - The cost of painting is 1 action.
    - The heuristic ignores potential conflicts (multiple robots wanting the same tile or path)
      and the effect of painting on tile clearance for future moves.

    # Heuristic Initialization
    - Extracts goal conditions to find target colors for tiles.
    - Builds an undirected adjacency graph of tiles based on static up/down/left/right facts
      to calculate movement distances using BFS.

    # Step-by-Step Thinking for Computing Heuristic
    1. Identify all tiles that need to be painted according to the goal and their target colors.
    2. For each goal tile that is *not* currently painted correctly:
       - Calculate the minimum cost for *any* robot to paint this tile.
       - The cost for a robot R to paint tile T with color C is:
         - Cost to get color C: 1 if R's current color is not C, 0 otherwise.
         - Cost to move to an adjacent tile: Minimum number of moves (BFS distance) from R's current location to *any* tile adjacent to T, traversing only through tiles that are currently *clear*.
         - Cost to paint: 1 action.
       - The minimum of these robot costs is the estimated cost for tile T.
    3. Sum the estimated costs for all unpainted goal tiles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal colors for each tile.
        - Adjacency graph of tiles from static facts.
        """
        super().__init__(task) # Call the base class constructor

        # Store goal colors for each tile {tile_name: color_name}
        self.goal_colors = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "painted":
                # Goal fact is (painted tile color)
                if len(args) == 2:
                    tile, color = args
                    self.goal_colors[tile] = color
                # else: Handle potential malformed goal facts if necessary

        # Build undirected adjacency graph {tile_name: [neighbor1, neighbor2, ...]}
        # Represents possible moves between adjacent tiles.
        self.adj = defaultdict(list)
        for fact in self.static:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] in ["up", "down", "left", "right"]:
                # Predicate is (direction tile1 tile2) meaning tile1 is direction of tile2
                # e.g., (up tile_1_1 tile_0_1) means tile_1_1 is up from tile_0_1.
                # A robot can move from tile_0_1 to tile_1_1 (move_up)
                # A robot can move from tile_1_1 to tile_0_1 (move_down)
                # So, there's an edge between tile1 and tile2 in both directions for BFS.
                tile1, tile2 = parts[1], parts[2]
                self.adj[tile1].append(tile2)
                self.adj[tile2].append(tile1)

        # Remove duplicates from adjacency lists (though defaultdict(list) and appending might not create duplicates if facts are unique)
        # This step ensures uniqueness just in case.
        for tile in self.adj:
            self.adj[tile] = list(set(self.adj[tile]))

    def _bfs_distance(self, start_tile, target_adj_tiles, state_clear_tiles):
        """
        Calculates the shortest path distance from start_tile to any tile
        in target_adj_tiles, only traversing through tiles in state_clear_tiles.
        Returns distance or infinity if unreachable.
        """
        # If there are no target adjacent tiles (e.g., goal tile is isolated), it's unreachable.
        if not target_adj_tiles:
            return float('inf')

        # BFS queue stores tuples of (current_tile, distance_from_start)
        queue = deque([(start_tile, 0)])
        # Keep track of visited tiles to avoid cycles and redundant work
        visited = {start_tile}

        while queue:
            curr_tile, dist = queue.popleft()

            # Check if the current tile is one of the target adjacent tiles
            if curr_tile in target_adj_tiles:
                return dist

            # Explore neighbors of the current tile
            for neighbor in self.adj.get(curr_tile, []):
                # A robot can only move onto a tile if it is clear.
                # The BFS path must only traverse through clear tiles.
                # The starting tile does not need to be clear to start *from* it.
                if neighbor in state_clear_tiles and neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

        # If the queue is empty and we haven't reached a target adjacent tile, it's unreachable
        return float('inf')

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach a goal state from the current state.
        """
        state = node.state

        # Extract current robot states {robot_name: {'location': tile, 'color': color}}
        robot_states = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "robot-at":
                robot, location = parts[1], parts[2]
                if robot not in robot_states:
                    robot_states[robot] = {}
                robot_states[robot]['location'] = location
            elif len(parts) == 3 and parts[0] == "robot-has":
                robot, color = parts[1], parts[2]
                if robot not in robot_states:
                    robot_states[robot] = {}
                robot_states[robot]['color'] = color
            # Assuming robot_states will be fully populated for all robots in the problem

        # Extract current painted tiles {tile_name: color_name}
        painted_state = {}
        for fact in state:
            parts = get_parts(fact)
            if len(parts) == 3 and parts[0] == "painted":
                tile, color = parts[1], parts[2]
                painted_state[tile] = color

        # Extract current clear tiles set
        state_clear_tiles = {get_parts(fact)[1] for fact in state if match(fact, "clear", "*")}

        total_heuristic_cost = 0

        # Iterate through each tile that needs to be painted according to the goal
        for goal_tile, goal_color in self.goal_colors.items():
            # Check if the tile is already painted correctly in the current state
            if goal_tile in painted_state and painted_state[goal_tile] == goal_color:
                continue # This goal fact is already satisfied

            # This tile needs to be painted with goal_color.
            # Calculate the minimum cost for any robot to achieve this.

            min_robot_cost_for_tile = float('inf')

            # Find tiles adjacent to the goal_tile using the pre-built adjacency graph
            target_adj_tiles = set(self.adj.get(goal_tile, []))

            # If a goal tile has no adjacent tiles defined in the static facts,
            # it's impossible to paint it using paint actions. This shouldn't
            # happen in valid instances, but check for robustness.
            if not target_adj_tiles:
                 # This tile cannot be painted. If it's a goal, the state is likely unsolvable.
                 # Returning infinity for the whole heuristic might be appropriate,
                 # but summing finite costs for other tiles might be more informative
                 # for greedy search if only *some* goals are unreachable.
                 # For now, let the min_robot_cost_for_tile remain infinity, which
                 # will propagate to the total if this is the only unpainted tile.
                 continue # Skip this tile if it has no adjacent tiles

            # Iterate through all robots to find the minimum cost for this specific tile
            for robot_name, robot_info in robot_states.items():
                # Ensure robot_info has both location and color (should be true based on domain init)
                if 'location' not in robot_info or 'color' not in robot_info:
                    continue # Skip if robot state is incomplete

                robot_location = robot_info['location']
                robot_color = robot_info['color']

                # Cost to get the correct color: 1 action if current color is wrong, 0 otherwise.
                color_cost = 0 if robot_color == goal_color else 1

                # Cost to move from the robot's current location to any tile adjacent to the goal_tile.
                # The path must only go through tiles that are currently clear.
                move_cost = self._bfs_distance(robot_location, target_adj_tiles, state_clear_tiles)

                # If no path exists to any adjacent tile (e.g., grid is blocked by non-clear tiles),
                # this robot cannot paint this tile.
                if move_cost == float('inf'):
                    continue # This robot cannot reach, try the next robot

                # Cost to paint the tile: 1 action (paint_up/down/left/right)
                paint_cost = 1

                # Total estimated cost for this robot to paint this specific tile
                robot_cost = color_cost + move_cost + paint_cost

                # Update the minimum cost found so far for this tile across all robots
                min_robot_cost_for_tile = min(min_robot_cost_for_tile, robot_cost)

            # Add the minimum cost for this tile to the total heuristic cost.
            # If min_robot_cost_for_tile is still infinity, it means no robot can reach
            # an adjacent clear tile to paint this goal tile. This might indicate
            # an unsolvable state or a limitation of the heuristic/BFS.
            # Adding infinity here makes the total cost infinity if any required tile
            # is unreachable. This is a reasonable behavior for a heuristic.
            if min_robot_cost_for_tile != float('inf'):
                 total_heuristic_cost += min_robot_cost_for_tile
            else:
                 # If even the best robot cannot reach a tile, this goal might be impossible
                 # in this state. Returning infinity might be appropriate for the whole state.
                 # However, summing finite costs for other goals might be more informative.
                 # Let's assume if min_robot_cost_for_tile is inf, it means this specific
                 # goal tile contributes infinity to the total, effectively making the
                 # total infinity if any required tile is unreachable.
                 total_heuristic_cost = float('inf')
                 break # No need to check other tiles if one is unreachable

        return total_heuristic_cost

# Example Usage (assuming you have a Task object and a Node object)
# from task import Task, Operator # Assuming Task and Operator classes are available

# # Example Task and Node creation (for testing purposes, not part of the final heuristic code)
# # This part is illustrative and depends on how your planner loads PDDL.
# if __name__ == "__main__":
#     # Minimal example setup (replace with actual PDDL parsing)
#     class MockTask:
#         def __init__(self, name, facts, initial_state, goals, operators, static):
#             self.name = name
#             self.facts = facts
#             self.initial_state = initial_state
#             self.goals = goals
#             self.operators = operators
#             self.static = static

#     class MockNode:
#         def __init__(self, state):
#             self.state = state

#     # Example 1 data (simplified)
#     task1_static = frozenset({'(available-color white)', '(available-color black)', '(up tile_1_1 tile_0_1)', '(down tile_0_1 tile_1_1)'})
#     task1_goals = frozenset({'(painted tile_1_1 white)'})
#     task1_initial_state = frozenset({'(robot-at robot1 tile_0_1)', '(robot-has robot1 black)', '(available-color white)', '(available-color black)', '(clear tile_1_1)'})
#     task1 = MockTask("floortile-01", None, task1_initial_state, task1_goals, None, task1_static)
#     node1 = MockNode(task1_initial_state)

#     # Example 2 data (simplified static/initial/goal for a few tiles)
#     task2_static = frozenset({
#         '(available-color white)', '(available-color black)',
#         '(up tile_1_1 tile_0_1)', '(down tile_0_1 tile_1_1)',
#         '(up tile_1_2 tile_0_2)', '(down tile_0_2 tile_1_2)',
#         '(left tile_0_1 tile_0_2)', '(right tile_0_2 tile_0_1)',
#         '(up tile_2_1 tile_1_1)', '(down tile_1_1 tile_2_1)',
#         '(left tile_1_1 tile_1_2)', '(right tile_1_2 tile_1_1)',
#         '(up tile_3_2 tile_2_2)', '(down tile_2_2 tile_3_2)',
#         '(left tile_2_2 tile_2_3)', '(right tile_2_3 tile_2_2)',
#         # Add more static facts to build a larger grid graph
#         '(up tile_2_2 tile_1_2)', '(down tile_1_2 tile_2_2)',
#         '(left tile_3_2 tile_3_3)', '(right tile_3_3 tile_3_2)',
#         '(up tile_2_4 tile_1_4)', '(down tile_1_4 tile_2_4)',
#         '(left tile_2_4 tile_2_5)', '(right tile_2_5 tile_2_4)',
#         '(up tile_2_6 tile_1_6)', '(down tile_1_6 tile_2_6)',
#         '(left tile_2_6 tile_2_5)', '(right tile_2_5 tile_2_6)',
#     })
#     task2_goals = frozenset({
#         '(painted tile_1_1 white)',
#         '(painted tile_1_2 black)',
#         '(painted tile_2_6 white)',
#     })
#     task2_initial_state = frozenset({
#         '(robot-at robot1 tile_3_2)', '(robot-has robot1 black)',
#         '(robot-at robot2 tile_2_6)', '(robot-has robot2 white)',
#         '(robot-at robot3 tile_2_4)', '(robot-has robot3 white)',
#         '(available-color white)', '(available-color black)',
#         # Tiles that are clear
#         '(clear tile_0_1)', '(clear tile_0_2)', '(clear tile_1_1)', '(clear tile_1_2)',
#         '(clear tile_2_1)', '(clear tile_2_2)', '(clear tile_2_3)', '(clear tile_2_5)',
#         '(clear tile_3_1)', '(clear tile_3_3)',
#         # Tiles that are NOT clear (robot locations or already painted - though none painted in this simplified init)
#         # Robot locations are implicitly not clear by action effects, but initial state might list them as clear.
#         # Let's assume the initial state correctly lists clear tiles *not* occupied by robots.
#         # So, tile_3_2, tile_2_6, tile_2_4 are NOT in state_clear_tiles based on the domain effects.
#         # The provided example state shows robot location *and* clear tiles, where the robot location is NOT clear.
#         # Let's use the provided example state logic.
#         '(clear tile_0_1)', '(clear tile_0_2)', '(clear tile_0_3)', '(clear tile_0_4)', '(clear tile_0_5)', '(clear tile_0_6)',
#         '(clear tile_1_1)', '(clear tile_1_2)', '(clear tile_1_3)', '(clear tile_1_4)', '(clear tile_1_5)', '(clear tile_1_6)',
#         '(clear tile_2_1)', '(clear tile_2_2)', '(clear tile_2_3)', '(clear tile_2_5)', # tile_2_4, tile_2_6 not clear
#         '(clear tile_3_1)', '(clear tile_3_3)', '(clear tile_3_4)', '(clear tile_3_5)', '(clear tile_3_6)', # tile_3_2 not clear
#         # ... add all clear tiles from example 2 init
#         '(robot-at robot1 tile_3_2)', '(robot-has robot1 black)',
#         '(robot-at robot2 tile_2_6)', '(robot-has robot2 white)',
#         '(robot-at robot3 tile_2_4)', '(robot-has robot3 white)',
#     })
#     task2 = MockTask("floortile-99", None, task2_initial_state, task2_goals, None, task2_static)
#     node2 = MockNode(task2_initial_state)


#     # Test Example 1
#     print("Testing Example 1:")
#     heuristic1 = floortileHeuristic(task1)
#     h_value1 = heuristic1(node1)
#     print(f"Heuristic value for Example 1: {h_value1}") # Expected: 2 (change color + paint)

#     # Test Example 2 (simplified goals)
#     print("\nTesting Example 2 (simplified goals):")
#     heuristic2 = floortileHeuristic(task2)
#     h_value2 = heuristic2(node2)
#     print(f"Heuristic value for Example 2 (simplified goals): {h_value2}")
#     # Let's manually trace h_value2 for tile_1_1 (white):
#     # Goal: (painted tile_1_1 white)
#     # Robots: R1@tile_3_2(black), R2@tile_2_6(white), R3@tile_2_4(white)
#     # Adjacent to tile_1_1: tile_0_1, tile_2_1, tile_1_2 (based on static)
#     # Clear tiles in state: tile_0_1, tile_0_2, tile_1_1, tile_1_2, tile_2_1, tile_2_2, tile_2_3, tile_2_5, tile_3_1, tile_3_3, ...
#     # R1@tile_3_2(black): Needs white (cost 1). BFS from tile_3_2 to {tile_0_1, tile_2_1, tile_1_2} through clear tiles.
#     # Path tile_3_2 -> tile_2_2 (clear) -> tile_1_2 (clear). Dist 2. Total R1 cost: 1 + 2 + 1 = 4.
#     # R2@tile_2_6(white): Has white (cost 0). BFS from tile_2_6 to {tile_0_1, tile_2_1, tile_1_2} through clear tiles.
#     # Path tile_2_6 -> tile_2_5 (clear) -> tile_2_3 (clear) -> tile_2_2 (clear) -> tile_2_1 (clear). Dist 4. Total R2 cost: 0 + 4 + 1 = 5.
#     # R3@tile_2_4(white): Has white (cost 0). BFS from tile_2_4 to {tile_0_1, tile_2_1, tile_1_2} through clear tiles.
#     # Path tile_2_4 -> tile_2_3 (clear) -> tile_2_2 (clear) -> tile_2_1 (clear). Dist 3. Total R3 cost: 0 + 3 + 1 = 4.
#     # Min cost for tile_1_1 is 4.

#     # Goal: (painted tile_1_2 black)
#     # Robots: R1@tile_3_2(black), R2@tile_2_6(white), R3@tile_2_4(white)
#     # Adjacent to tile_1_2: tile_0_2, tile_2_2, tile_1_1, tile_1_3 (based on static)
#     # Clear tiles: ...
#     # R1@tile_3_2(black): Has black (cost 0). BFS from tile_3_2 to {tile_0_2, tile_2_2, tile_1_1, tile_1_3} through clear.
#     # Path tile_3_2 -> tile_2_2 (clear). Dist 1. Total R1 cost: 0 + 1 + 1 = 2.
#     # R2@tile_2_6(white): Needs black (cost 1). BFS from tile_2_6 to {tile_0_2, tile_2_2, tile_1_1, tile_1_3} through clear.
#     # Path tile_2_6 -> tile_2_5 (clear) -> tile_2_3 (clear) -> tile_2_2 (clear). Dist 3. Total R2 cost: 1 + 3 + 1 = 5.
#     # R3@tile_2_4(white): Needs black (cost 1). BFS from tile_2_4 to {tile_0_2, tile_2_2, tile_1_1, tile_1_3} through clear.
#     # Path tile_2_4 -> tile_2_3 (clear) -> tile_2_2 (clear). Dist 2. Total R3 cost: 1 + 2 + 1 = 4.
#     # Min cost for tile_1_2 is 2.

#     # Goal: (painted tile_2_6 white)
#     # Robots: R1@tile_3_2(black), R2@tile_2_6(white), R3@tile_2_4(white)
#     # Adjacent to tile_2_6: tile_1_6, tile_3_6, tile_2_5 (based on static)
#     # Clear tiles: ... tile_2_5, tile_3_6, tile_1_6 (assuming these are clear in the state)
#     # R1@tile_3_2(black): Needs white (cost 1). BFS from tile_3_2 to {tile_1_6, tile_3_6, tile_2_5} through clear.
#     # Path tile_3_2 -> tile_3_3 (clear) -> tile_3_4 (clear) -> tile_3_5 (clear) -> tile_3_6 (clear). Dist 4. Total R1 cost: 1 + 4 + 1 = 6.
#     # R2@tile_2_6(white): Has white (cost 0). Robot is *at* tile_2_6. Need to move to an *adjacent* tile.
#     # BFS from tile_2_6 to {tile_1_6, tile_3_6, tile_2_5} through clear.
#     # Robot is at tile_2_6, which is adjacent to tile_2_5. Distance 1 (move to tile_2_5).
#     # Note: The BFS starts from robot_location. If robot_location is adjacent to target, dist is 1.
#     # The BFS is looking for distance from start_tile to *any* tile in target_adj_tiles.
#     # If start_tile is tile_2_6 and target_adj_tiles includes tile_2_5, the BFS will find tile_2_5 as a neighbor at dist 1.
#     # Total R2 cost: 0 + 1 + 1 = 2.
#     # R3@tile_2_4(white): Has white (cost 0). BFS from tile_2_4 to {tile_1_6, tile_3_6, tile_2_5} through clear.
#     # Path tile_2_4 -> tile_2_5 (clear). Dist 1. Total R3 cost: 0 + 1 + 1 = 2.
#     # Min cost for tile_2_6 is 2.

#     # Total heuristic for simplified goals: 4 (for tile_1_1) + 2 (for tile_1_2) + 2 (for tile_2_6) = 8.
#     # The actual output might differ slightly based on the full set of clear tiles in the state.
#     # The BFS needs the *actual* clear tiles from the state object.
#     # Let's re-check the provided state example:
#     # frozenset({'(clear tile_1_5)', '(clear tile_3_1)', '(clear tile_3_3)', '(clear tile_3_4)', '(clear tile_0_1)', '(clear tile_0_2)', '(clear tile_0_3)', '(clear tile_3_2)', '(painted tile_1_2 black)', '(painted tile_1_3 white)', '(painted tile_1_4 black)', '(painted tile_3_5 white)', '(painted tile_1_1 white)', '(clear tile_2_1)', '(clear tile_2_2)', '(clear tile_2_3)', '(clear tile_2_4)', '(clear tile_2_5)', '(painted tile_3_6 black)', '(painted tile_2_6 white)', '(painted tile_1_6 black)', '(clear tile_0_6)', '(robot-has robot1 white)', '(robot-at robot1 tile_0_4)', '(clear tile_0_5)'})
#     # This state has Robot1 at tile_0_4 with white.
#     # It has several tiles already painted: tile_1_2 (black), tile_1_3 (white), tile_1_4 (black), tile_3_5 (white), tile_1_1 (white), tile_3_6 (black), tile_2_6 (white), tile_1_6 (black).
#     # Let's assume the goal is the full goal from instance-file-example-2.
#     # Goal: (painted tile_1_1 white) - Achieved in this state. Cost 0.
#     # Goal: (painted tile_1_2 black) - Achieved. Cost 0.
#     # Goal: (painted tile_1_3 white) - Achieved. Cost 0.
#     # Goal: (painted tile_1_4 black) - Achieved. Cost 0.
#     # Goal: (painted tile_1_5 white) - NOT achieved. Needs white. Tile_1_5 is clear.
#     # Goal: (painted tile_1_6 black) - Achieved. Cost 0.
#     # Goal: (painted tile_2_1 black) - NOT achieved. Needs black. Tile_2_1 is clear.
#     # Goal: (painted tile_2_2 white) - NOT achieved. Needs white. Tile_2_2 is clear.
#     # Goal: (painted tile_2_3 black) - NOT achieved. Needs black. Tile_2_3 is clear.
#     # Goal: (painted tile_2_4 white) - NOT achieved. Needs white. Tile_2_4 is clear.
#     # Goal: (painted tile_2_5 black) - NOT achieved. Needs black. Tile_2_5 is clear.
#     # Goal: (painted tile_2_6 white) - Achieved. Cost 0.
#     # ... and so on for tiles 3, 4, 5.

#     # Let's calculate for tile_1_5 (needs white) in the provided state:
#     # Robot1@tile_0_4(white). Tile_1_5 needs white. Robot has white (cost 0).
#     # Adjacent to tile_1_5: tile_0_5, tile_2_5, tile_1_4, tile_1_6.
#     # Clear tiles in state: tile_1_5, tile_3_1, tile_3_3, tile_3_4, tile_0_1, tile_0_2, tile_0_3, tile_3_2, tile_2_1, tile_2_2, tile_2_3, tile_2_4, tile_2_5, tile_0_6, tile_0_5.
#     # Note: tile_1_4 and tile_1_6 are NOT clear (painted).
#     # Adjacent clear tiles to tile_1_5: tile_0_5, tile_2_5.
#     # BFS from tile_0_4 to {tile_0_5, tile_2_5} through clear tiles.
#     # Path tile_0_4 -> tile_0_5 (clear). Dist 1.
#     # Min move cost is 1.
#     # Total cost for tile_1_5: 0 (color) + 1 (move) + 1 (paint) = 2.

#     # This trace confirms the logic. The heuristic sums these minimum costs for all unpainted goal tiles.
#     # The actual heuristic value for the provided state and full goal would be the sum of costs for
#     # tile_1_5, tile_2_1, tile_2_2, tile_2_3, tile_2_4, tile_2_5, plus all unpainted tiles in rows 3, 4, 5.
#     # This requires running the code with the full goal and state.
#     # The manual trace for tile_1_5 gives 2.
#     # The manual trace for tile_1_1, tile_1_2, tile_2_6 in the simplified example gave 4, 2, 2.
#     # The logic seems consistent.
