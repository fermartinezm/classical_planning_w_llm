from fnmatch import fnmatch
from collections import deque, defaultdict
# Assuming heuristic_base is available in the environment
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Example: "(at-robot loc_1_1)" -> ["at-robot", "loc_1_1"]
    return fact[1:-1].split()

# Helper function to match PDDL facts (optional, but good practice)
# Not strictly needed for this specific heuristic, but useful in general.
# def match(fact, *args):
#     """
#     Check if a PDDL fact matches a given pattern.
#     - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
#     - `args`: The expected pattern (wildcards `*` allowed).
#     - Returns `True` if the fact matches the pattern, else `False`.
#     """
#     parts = get_parts(fact)
#     return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing the
    minimum number of pushes required for each box to reach its goal location
    and the minimum number of robot moves required to reach any ungoaled box.
    It is designed for greedy best-first search and is not admissible.

    # Assumptions
    - The grid structure and traversable paths are defined solely by 'adjacent' predicates.
    - All locations mentioned in 'adjacent' facts, initial state, and goal state are relevant.
    - All locations within a connected component of the graph are reachable from each other.
    - Each box relevant to the goal has a unique goal location specified in the problem goal.
    - States where a box cannot reach its goal or the robot cannot reach an ungoaled box are considered effectively infinite cost.

    # Heuristic Initialization
    - Parses goal facts to create a mapping from each box object to its target goal location.
    - Builds an undirected graph where nodes are locations and edges represent 'adjacent' relationships from the static facts.
    - Computes all-pairs shortest paths between all locations in the graph using Breadth-First Search (BFS). These distances represent the minimum number of moves (robot or box) between any two locations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and the current location of every box that has a specified goal location.
    2. Determine which of these boxes are not currently at their respective goal locations. These are the "ungoaled" boxes.
    3. If there are no ungoaled boxes, the state is a goal state, and the heuristic value is 0.
    4. If there are ungoaled boxes, initialize the total heuristic value `h` to 0.
    5. For each ungoaled box:
       - Get its current location and its target goal location.
       - Look up the precomputed shortest path distance between the box's current location and its goal location using the graph distances. This distance represents the minimum number of 'steps' or pushes required for this box to reach its goal, ignoring obstacles other than the grid structure itself.
       - Add this distance to `h`.
       - If the goal location is not reachable from the box's current location within the graph (distance is not found), the state is likely unsolvable; return a large value (infinity).
    6. Calculate the minimum shortest path distance from the robot's current location to the current location of *any* of the ungoaled boxes. This estimates the cost for the robot to reach a box it needs to push.
    7. Add this minimum robot-to-box distance to `h`.
    8. If the robot's location or any ungoaled box's location is not in the graph, or if an ungoaled box is unreachable by the robot, the state is likely unsolvable; return a large value (infinity).
    9. Return the total calculated heuristic value `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building
        the location graph for distance calculations.

        Args:
            task: An object representing the planning task, containing goals and static facts.
        """
        # task is an instance of the Task class provided by the planner
        self.goals = task.goals
        self.static = task.static

        # 1. Parse goal facts to get box-goal mappings
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at box_name loc_name)
            if parts[0] == "at" and len(parts) == 3:
                box, location = parts[1], parts[2]
                self.goal_locations[box] = location
            # Note: Assumes goal is a set of atomic facts like (at box loc).
            # If it could be (and (...)), recursive parsing would be needed.

        # 2. Build the location graph from 'adjacent' facts
        self.graph = defaultdict(list)
        self.all_locations = set()
        for fact in self.static:
            parts = get_parts(fact)
            # Adjacent facts are typically (adjacent loc1 loc2 dir)
            if parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2 = parts[1], parts[2]
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1) # Graph is undirected for movement
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Add any locations mentioned in goals that might not be in adjacent facts
        # This ensures goal locations are nodes in our graph, even if isolated.
        for goal_loc in self.goal_locations.values():
             self.all_locations.add(goal_loc)

        # 3. Compute all-pairs shortest paths using BFS
        self.dist = {}
        for start_node in self.all_locations:
            self.dist[start_node] = self._bfs(start_node)

    def _bfs(self, start_node):
        """
        Performs BFS from a start node to find distances to all reachable nodes
        within the graph.

        Args:
            start_node: The location string to start BFS from.

        Returns:
            A dictionary mapping reachable location strings to their shortest
            distance from the start_node.
        """
        distances = {start_node: 0}
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()
            distance = distances[current_node]

            # Get neighbors from the graph, handle nodes not in graph keys gracefully
            # A node might be in all_locations but not have any adjacent facts
            for neighbor in self.graph.get(current_node, []):
                if neighbor not in distances:
                    distances[neighbor] = distance + 1
                    queue.append(neighbor)
        return distances

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to reach
        a goal state from the current node's state.

        Args:
            node: The current node in the search tree, containing the state.

        Returns:
            An estimated cost (integer) or float('inf') if the state is likely
            unsolvable or malformed.
        """
        state = node.state  # frozenset of facts

        # 1. Identify current locations of robot and relevant boxes
        robot_location = None
        box_locations = {} # Only store locations for boxes relevant to goals
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot" and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == "at" and len(parts) == 3 and parts[1] in self.goal_locations:
                 box_locations[parts[1]] = parts[2]

        # Basic check for expected state elements
        # If robot or any box with a goal is missing, the state is malformed.
        if robot_location is None or len(box_locations) != len(self.goal_locations):
             return float('inf')

        # 2. Identify ungoaled boxes
        ungoaled_boxes = []
        for box, goal_loc in self.goal_locations.items():
            if box_locations[box] != goal_loc:
                ungoaled_boxes.append(box)

        # 3. If all boxes are at goals, heuristic is 0
        if not ungoaled_boxes:
            return 0

        # 4. Initialize heuristic value
        h = 0

        # 5. Add box-goal distances for each ungoaled box
        for box in ungoaled_boxes:
            current_loc = box_locations[box]
            goal_loc = self.goal_locations[box]

            # Check if locations are in our precomputed distances and reachable
            # If goal is unreachable from box, state is likely unsolvable.
            if current_loc not in self.dist or goal_loc not in self.dist[current_loc]:
                 return float('inf')

            h += self.dist[current_loc][goal_loc]

        # 6. Find minimum robot-to-box distance for any ungoaled box
        min_robot_box_dist = float('inf')

        # Check if robot's current location is in our graph and can reach locations
        if robot_location not in self.dist:
             # Robot is in an isolated part of the graph or malformed state
             return float('inf')

        for box in ungoaled_boxes:
            box_loc = box_locations[box]

            # Check if box location is reachable by the robot
            if box_loc not in self.dist[robot_location]:
                 # An ungoaled box is unreachable by the robot
                 return float('inf')

            min_robot_box_dist = min(min_robot_box_dist, self.dist[robot_location][box_loc])

        # 7. Add minimum robot-to-box distance
        h += min_robot_box_dist

        # 8. Return total heuristic value
        return h
