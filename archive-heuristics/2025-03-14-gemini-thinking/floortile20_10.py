from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
import collections

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(painted tile_1_1 white)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the floortile domain.

    # Summary
    This heuristic estimates the number of actions required to reach the goal state in the floortile domain.
    It focuses on the number of tiles that are not yet painted according to the goal specification,
    and considers the necessity of color changes and robot movements to reach and paint those tiles.

    # Assumptions:
    - The primary actions are painting tiles and moving the robot.
    - Color changes are also necessary actions.
    - The heuristic prioritizes painting unpainted goal tiles and accounts for the cost of movement and color changes.

    # Heuristic Initialization
    - Extracts the goal conditions, specifically the `painted` predicates, to determine which tiles need to be painted with what color.
    - Preprocesses the static facts to efficiently access tile adjacency information (up, down, left, right relations).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the goal tiles and their target colors from the goal conditions.
    2. For each goal tile that is not yet painted with the correct color in the current state:
        a. Increment the heuristic cost by 1, representing the 'paint' action needed.
        b. Determine the color required for the goal tile.
        c. Check if the robot currently holds the correct color. If not, increment the heuristic cost by 1, representing a 'change_color' action.
        d. Find the current location of the robot.
        e. Calculate the shortest path (in terms of number of move actions) from the robot's current location to the unpainted goal tile. This is done using Breadth-First Search (BFS) on the tile grid, considering the 'clear' predicate as a constraint for movement. Add the length of this shortest path to the heuristic cost.
    3. Sum up the costs for all unpainted goal tiles to get the total heuristic estimate.
    """

    def __init__(self, task):
        """
        Initialize the floortile heuristic.

        Extracts goal conditions and static facts to prepare for heuristic calculations.
        Specifically, it focuses on 'painted' goal predicates and tile adjacency relations.
        """
        self.goals = task.goals
        static_facts = task.static

        self.goal_tiles_colors = {}
        for goal in self.goals:
            if match(goal, "painted", "*", "*"):
                parts = get_parts(goal)
                tile = parts[1]
                color = parts[2]
                self.goal_tiles_colors[tile] = color

        self.adjacency_list = collections.defaultdict(list)
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] in ["up", "down", "left", "right"]:
                tile1 = parts[2]
                tile2 = parts[1] # Note the order in static facts is (direction tile2 tile1)
                self.adjacency_list[tile1].append((parts[0], tile2))


    def __call__(self, node):
        """
        Calculate the heuristic value for a given state.

        Estimates the number of actions needed to reach the goal state from the current state
        based on the number of unpainted goal tiles, required color changes, and robot movements.
        """
        state = node.state
        heuristic_value = 0

        robot_location = None
        robot_color = None
        clear_tiles = set()
        painted_tiles = {}

        for fact in state:
            if match(fact, "robot-at", "*", "*"):
                robot_location = get_parts(fact)[2]
            elif match(fact, "robot-has", "*", "*"):
                robot_color = get_parts(fact)[2]
            elif match(fact, "clear", "*"):
                clear_tiles.add(get_parts(fact)[1])
            elif match(fact, "painted", "*", "*"):
                painted_tiles[get_parts(fact)[1]] = get_parts(fact)[2]

        if not self.goal_tiles_colors: # if no goal tiles, heuristic is 0
            return 0

        unpainted_goal_tiles = []
        for tile, goal_color in self.goal_tiles_colors.items():
            if tile not in painted_tiles or painted_tiles[tile] != goal_color:
                unpainted_goal_tiles.append((tile, goal_color))

        if not unpainted_goal_tiles: # if all goal tiles are painted, heuristic is 0
            return 0


        for goal_tile, goal_color in unpainted_goal_tiles:
            heuristic_value += 1 # for paint action

            if robot_color != goal_color:
                heuristic_value += 1 # for change_color action

            if robot_location:
                queue = collections.deque([(robot_location, 0)])
                visited = {robot_location}
                path_found = False
                shortest_path = float('inf')

                while queue:
                    current_tile, distance = queue.popleft()
                    if current_tile == goal_tile:
                        shortest_path = distance
                        path_found = True
                        break

                    for direction, neighbor_tile in self.adjacency_list[current_tile]:
                        if neighbor_tile in clear_tiles and neighbor_tile not in visited:
                            visited.add(neighbor_tile)
                            queue.append((neighbor_tile, distance + 1))
                if path_found:
                    heuristic_value += shortest_path
                else:
                    heuristic_value += 100 # large penalty if goal tile is unreachable (should not happen in well-formed problems)


        return heuristic_value
