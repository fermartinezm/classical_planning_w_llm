from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at robot1 tile_0_0)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class floortileHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Floortile domain.

    # Summary
    This heuristic estimates the number of actions required to paint all tiles
    that are not currently painted correctly according to the goal. It sums
    the estimated costs for painting each required tile, considering the paint
    action itself, the need to change color, and the movement required to get
    adjacent to the tile.

    # Assumptions
    - All actions have a cost of 1.
    - Tiles that need painting are currently 'clear'. Tiles painted with the
      wrong color cannot be repainted (based on domain actions).
    - The grid structure is defined by 'up', 'down', 'left', 'right' predicates.
    - The robot is the only entity that can paint.
    - There is only one robot, assumed to be named 'robot1'.

    # Heuristic Initialization
    - Extract goal conditions to identify which tiles need to be painted and with which color.
    - Parse static facts to build the grid graph (adjacency list) representing tile connectivity.
    - Compute all-pairs shortest paths between all tiles on the grid using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the set of tiles that are required to be painted in the goal state
       but are not currently painted correctly in the given state. Let this set
       of unsatisfied goal facts be `G_unsat`.
    2. If `G_unsat` is empty, the state is a goal state, and the heuristic is 0.
    3. Initialize the total estimated cost to 0.
    4. Add the cost for painting each unsatisfied tile: Add `|G_unsat|` to the total cost.
    5. Identify the set of distinct colors required for the tiles in `G_unsat`.
    6. Determine the robot's current color by finding the `(robot-has robot1 color)` fact.
    7. Estimate the color change cost: Count the number of distinct required colors. If the robot
       already has one of these colors, subtract 1 from the count (as the first batch
       of tiles can be painted without an initial color change). Add this value to the total cost.
       If no tiles need painting, the color cost is 0.
    8. Determine the robot's current location by finding the `(robot-at robot1 tile)` fact.
    9. Estimate the movement cost: For each tile `t` that needs painting (i.e., for each fact `(painted t c)` in `G_unsat`), calculate the minimum number of moves required for the robot to get from its current location to *any* tile adjacent to `t`. This minimum distance is `max(0, dist(robot_location, t) - 1)`, where `dist` is the shortest path distance on the grid. Sum these minimum distances for all tiles needing painting and add this sum to the total cost. This part is a simplification and likely overestimates movement, contributing to non-admissibility but aiming for better guidance. If the robot's location is unknown or a tile needing paint is unreachable, return infinity.
    10. Return the total estimated cost.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the grid graph."""
        self.goals = task.goals
        self.static = task.static

        # Extract all tile names and build adjacency list from static facts
        self.all_tiles = set()
        self.adjacency_list = {} # Graph representation: tile -> list of adjacent tiles

        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] in ['up', 'down', 'left', 'right']:
                # Connectivity facts are like (direction tile1 tile2)
                # where tile1 and tile2 are adjacent.
                t1, t2 = parts[1], parts[2]
                self.all_tiles.add(t1)
                self.all_tiles.add(t2)
                self.adjacency_list.setdefault(t1, []).append(t2)
                self.adjacency_list.setdefault(t2, []).append(t1) # Grid is undirected

        # Compute all-pairs shortest paths using BFS
        self.distances = {} # distances[t1][t2] = shortest_path_distance(t1, t2)

        for start_tile in self.all_tiles:
            self.distances[start_tile] = {}
            queue = [(start_tile, 0)]
            visited = {start_tile}

            head = 0 # Use index for queue for efficiency
            while head < len(queue):
                current_tile, d = queue[head]
                head += 1

                self.distances[start_tile][current_tile] = d

                # Ensure current_tile is in adjacency_list (might be isolated in weird problems)
                if current_tile in self.adjacency_list:
                    for neighbor in self.adjacency_list[current_tile]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append((neighbor, d + 1))

        # Store goal facts related to painting for quick lookup
        self.goal_painted_facts = {
            goal for goal in self.goals if match(goal, "painted", "*", "*")
        }


    def get_distance(self, tile1, tile2):
        """Returns the precomputed shortest path distance between two tiles."""
        # Use precomputed distance if available
        if tile1 in self.distances and tile2 in self.distances[tile1]:
             return self.distances[tile1][tile2]

        # This case should ideally not be reached in a well-formed problem
        # where all tiles are part of the connected grid defined by static facts.
        # Returning infinity indicates these tiles are unreachable from each other.
        # print(f"Warning: Distance between {tile1} and {tile2} not precomputed. Are they disconnected?")
        return float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify unsatisfied goal conditions related to painting
        unsat_painted_goals = {
            goal for goal in self.goal_painted_facts if goal not in state
        }

        # 2. If G_unsat is empty, heuristic is 0.
        if not unsat_painted_goals:
            return 0

        total_cost = 0

        # 3. Count paint actions
        total_cost += len(unsat_painted_goals)

        # 4. Identify distinct colors needed and tiles to paint
        colors_needed = set()
        tiles_to_paint = set()
        for goal_fact in unsat_painted_goals:
            _, tile, color = get_parts(goal_fact)
            colors_needed.add(color)
            tiles_to_paint.add(tile)

        # 5. Get robot's current color
        robot_color = None
        # Assuming only one robot named 'robot1' based on domain/examples
        robot_has_fact = next((fact for fact in state if match(fact, "robot-has", "robot1", "*")), None)
        if robot_has_fact:
             _, _, robot_color = get_parts(robot_has_fact)


        # 6. Calculate color change cost
        color_cost = 0
        if colors_needed: # Only add color cost if there are tiles to paint
            color_cost = len(colors_needed)
            if robot_color in colors_needed:
                color_cost -= 1 # Save one change if robot already has a needed color
            # If robot_color is None (e.g., free-color), it still needs to acquire the first color
            # The logic `robot_color in colors_needed` handles this correctly if None is not in the set.

        total_cost += color_cost

        # 7. Get robot's current location
        robot_location = None
        # Assuming only one robot named 'robot1'
        robot_at_fact = next((fact for fact in state if match(fact, "robot-at", "robot1", "*")), None)
        if robot_at_fact:
             _, _, robot_location = get_parts(robot_at_fact)


        # 8. Calculate movement cost
        move_cost = 0
        # Check if robot location is known and is a valid tile in our graph
        if robot_location and robot_location in self.all_tiles:
            for tile in tiles_to_paint:
                # Minimum distance to get adjacent to the tile
                # This is max(0, dist(robot_location, tile) - 1)
                dist_to_tile = self.get_distance(robot_location, tile)
                if dist_to_tile != float('inf'): # Only add cost if tile is reachable
                    move_cost += max(0, dist_to_tile - 1)
                else:
                    # If a tile needing paint is unreachable, the problem is likely unsolvable
                    # Returning infinity guides the search away from this path
                    return float('inf')
        else:
             # This case indicates an invalid state representation (robot not located or off-grid)
             # print(f"Warning: Robot location {robot_location} not found or not in grid tiles.")
             return float('inf') # Cannot solve without robot or if robot is off-grid

        total_cost += move_cost

        # 10. Return the total estimated cost
        return total_cost
