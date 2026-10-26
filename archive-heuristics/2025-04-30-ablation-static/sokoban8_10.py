from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at-robot loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the grid
    distance of each box to its goal location and adding the grid distance
    from the robot to the closest box that is not yet at its goal.

    # Assumptions
    - The grid structure is defined by the `adjacent` facts.
    - Each box has a unique goal location specified in the task goals.
    - The grid graph is connected (or at least, relevant locations are connected).

    # Heuristic Initialization
    - Extracts the goal location for each box from the task goals.
    - Builds a graph of locations based on `adjacent` facts.
    - Computes all-pairs shortest paths (grid distances) between all locations
      using Breadth-First Search (BFS).

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of the robot and all boxes from the state.
    2. For each box:
       - If the box is not at its designated goal location:
         - Calculate the grid distance from the box's current location to its goal location using precomputed distances.
         - Add this distance to a running total for box distances.
         - Calculate the grid distance from the robot's current location to this box's location using precomputed distances.
         - Keep track of the minimum robot-to-box distance among all off-goal boxes.
    3. If all boxes are at their goal locations, the heuristic is 0.
    4. Otherwise, the heuristic value is the sum of all box-to-goal distances plus the minimum robot-to-box distance. If the robot cannot reach any off-goal box or any off-goal box cannot reach its goal, the heuristic returns infinity.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for distance calculations.
        """
        # Assuming task object has attributes: goals, static
        self.goals = task.goals  # Goal conditions (frozenset of strings)
        self.static_facts = task.static  # Static facts (frozenset of strings)

        # Store goal locations for each box.
        self.box_goals = {}
        for goal in self.goals:
            # Goal facts are typically (at boxX loc_Y_Z)
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                box, location = args
                self.box_goals[box] = location

        # Build the adjacency graph from static facts.
        self.adjacencies = {}
        self.all_locations = set()
        for fact in self.static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)
                self.adjacencies.setdefault(loc1, []).append(loc2)
                self.adjacencies.setdefault(loc2, []).append(loc1) # Adjacency is symmetric

        # Compute all-pairs shortest paths using BFS.
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = self._bfs(start_loc)

    def _bfs(self, start_node):
        """
        Performs BFS starting from start_node to find distances to all other nodes.
        Returns a dictionary {node: distance}.
        """
        distances = {node: float('inf') for node in self.all_locations}
        distances[start_node] = 0
        queue = deque([start_node])
        visited = {start_node}

        while queue:
            current_node = queue.popleft()

            # Check if current_node has any adjacencies defined
            if current_node in self.adjacencies:
                for neighbor in self.adjacencies[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions.
        """
        state = node.state  # Current world state (frozenset of strings).

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        # If robot_loc is None, the state is likely invalid or the problem is malformed.
        # Cannot solve without a robot.
        if robot_loc is None:
             return float('inf')

        # Find box locations
        box_locations = {}
        for fact in state:
            # Check for facts like (at box1 loc_X_Y)
            if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 box_name = parts[1]
                 location_name = parts[2]
                 # Verify if the first argument is one of the boxes we care about
                 if box_name in self.box_goals:
                     box_locations[box_name] = location_name

        total_box_distance = 0
        min_robot_to_box_distance = float('inf')
        all_boxes_at_goal = True

        # Iterate through all boxes defined in the goals
        for box, goal_loc in self.box_goals.items():
            current_loc = box_locations.get(box)

            # If a box is not found in the current state, it's an invalid state
            # or the problem is malformed. Cannot solve.
            if current_loc is None:
                 return float('inf')

            if current_loc != goal_loc:
                all_boxes_at_goal = False

                # Get box-to-goal distance
                # Use .get() with {} as default to handle cases where current_loc or goal_loc
                # might not be in the precomputed distances (e.g., disconnected graph components)
                box_dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))
                if box_dist == float('inf'):
                    # Box goal is unreachable from its current location in the static graph.
                    # Problem is unsolvable from this state.
                    return float('inf')
                total_box_distance += box_dist

                # Get robot-to-box distance
                robot_to_this_box_dist = self.distances.get(robot_loc, {}).get(current_loc, float('inf'))
                # We don't return inf immediately here, as the robot might be able to reach *another* box.
                # We update min_robot_to_box_distance and check if it's still inf after the loop.
                min_robot_to_box_distance = min(min_robot_to_box_distance, robot_to_this_box_dist)


        if all_boxes_at_goal:
            return 0

        # If there are off-goal boxes but the robot cannot reach *any* of them
        # (min_robot_to_box_distance is still infinity), the state is unsolvable.
        if min_robot_to_box_distance == float('inf'):
             return float('inf')

        # Otherwise, return the sum of box distances and the minimum robot distance.
        return total_box_distance + min_robot_to_box_distance
