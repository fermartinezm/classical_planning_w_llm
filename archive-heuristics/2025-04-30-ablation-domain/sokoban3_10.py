from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at ball1 rooma)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing
    the estimated cost for each box that is not yet at its goal location.
    The estimated cost for a single box includes the minimum number of pushes
    required to move the box to its goal, plus the cost for the robot to
    reach the box's current location. This heuristic is non-admissible and
    designed to guide a greedy best-first search.

    # Assumptions
    - The grid structure and traversable locations are defined solely by
      'adjacent' facts in the static information. These facts form an
      undirected graph where movement is possible between adjacent locations.
    - Shortest path distances on this grid graph are used to estimate movement costs.
    - Each box in the problem has a unique, fixed goal location specified in the task goals.
    - The cost of moving a box is primarily the number of pushes (equal to the
      shortest path distance the box needs to travel).
    - The cost for the robot is estimated by the distance from the robot's
      current position to the box's current position. Robot repositioning
      costs between pushes for the same box are not explicitly modeled per push
      but are implicitly covered by summing the robot-to-box distance for each box.

    # Heuristic Initialization
    - Parses the task's goal conditions to create a mapping from each box
      object to its required goal location string.
    - Builds an undirected graph representation of the Sokoban grid based on
      the 'adjacent' facts provided in the task's static information. This graph
      is used for computing shortest path distances between locations.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state `s`:
    1.  Identify the robot's current location from the state facts (e.g., `(at-robot loc_X_Y)`).
    2.  Identify the current location of each box that has a goal location defined
        in the task. This is done by looking for facts like `(at box_N loc_A_B)` in the state.
    3.  Determine the set of boxes that are not currently at their designated goal locations.
    4.  If this set of boxes is empty, the state is a goal state, and the heuristic value is 0.
    5.  If there are boxes not at their goals, initialize a total heuristic cost to 0.
    6.  For each box `B` that is not at its goal:
        a.  Let `L_B` be the current location of box `B` and `G_B` be its goal location.
        b.  Calculate the shortest path distance `dist(L_B, G_B)` on the grid graph. This represents
            the minimum number of push actions required for box `B`. Add this distance to the total cost.
        c.  Calculate the shortest path distance `dist(Robot_Loc, L_B)` on the grid graph, where
            `Robot_Loc` is the robot's current location. This estimates the cost for the robot
            to reach box `B` to begin pushing it. Add this distance to the total cost.
        d.  If any required distance calculation (`dist(L_B, G_B)` or `dist(Robot_Loc, L_B)`)
            returns infinity (meaning the locations are disconnected in the graph), the state
            is likely unsolvable or very far from the goal; return infinity for the heuristic.
    7.  The final heuristic value is the sum accumulated in step 6. This sum represents
        an overestimation because the robot's travel cost is counted independently for
        each box, but it serves as a useful non-admissible estimate.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations and building the grid graph.

        Args:
            task: The planning task object containing initial state, goals, operators, and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Store goal locations for each box.
        # We assume any object appearing in an 'at' predicate in the goal is a box.
        self.box_goals = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2:
                obj, location = args[0], args[1]
                self.box_goals[obj] = location

        # Build the grid graph from adjacent facts
        # The graph is undirected as adjacency is symmetric (e.g., adjacent loc1 loc2 right implies adjacent loc2 loc1 left)
        self.graph = {}  # location -> set of adjacent locations
        self.locations = set() # Keep track of all known locations

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2, direction = parts[1], parts[2], parts[3]
                if l1 not in self.graph:
                    self.graph[l1] = set()
                if l2 not in self.graph:
                    self.graph[l2] = set()
                self.graph[l1].add(l2)
                self.graph[l2].add(l1) # Add reverse edge assuming symmetry
                self.locations.add(l1)
                self.locations.add(l2)

    def _get_distance(self, start_loc, end_loc):
        """
        Computes the shortest path distance between two locations using BFS on the grid graph.
        Returns float('inf') if end_loc is unreachable from start_loc or if locations are invalid.
        """
        if start_loc == end_loc:
            return 0
        # Ensure locations exist in the graph (handle potential parsing issues or isolated locations)
        if start_loc not in self.graph or end_loc not in self.graph:
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            # Check if current_loc has neighbors in the graph (should always be true if in self.graph)
            if current_loc in self.graph:
                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return float('inf') # end_loc is unreachable from start_loc

    def __call__(self, node):
        """
        Compute the heuristic estimate for the given state.

        Args:
            node: The search node containing the current state.

        Returns:
            An estimated cost (integer or float('inf')) to reach a goal state.
        """
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break

        # If robot location is not found, the state is likely invalid or terminal
        if robot_loc is None:
             return float('inf')

        # Find box locations for boxes that have goals
        box_locs = {}
        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is (at ?obj ?loc) and ?obj is one of the boxes we track
            if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj in self.box_goals:
                     box_locs[obj] = loc

        # Identify boxes not at their goal
        boxes_to_move = {
            box for box in self.box_goals.keys() # Iterate over all boxes with goals
            if box not in box_locs or box_locs[box] != self.box_goals[box]
        }

        # If all boxes are at their goal, heuristic is 0
        if not boxes_to_move:
            return 0

        total_heuristic = 0

        for box in boxes_to_move:
            # If a box with a goal is not found in the state's 'at' facts,
            # it might be inside a vehicle in a different domain, but in Sokoban
            # this likely means an invalid state or the box is lost.
            # Assuming boxes are always 'at' a location in Sokoban states.
            if box not in box_locs:
                 return float('inf') # Should not happen in valid Sokoban states

            current_box_loc = box_locs[box]
            goal_box_loc = self.box_goals[box]

            # Cost for the box to reach its goal (minimum pushes)
            box_dist = self._get_distance(current_box_loc, goal_box_loc)
            if box_dist == float('inf'):
                # Box cannot reach its goal from its current position
                return float('inf') # This state is likely unsolvable

            total_heuristic += box_dist

            # Cost for the robot to reach the box
            robot_to_box_dist = self._get_distance(robot_loc, current_box_loc)
            if robot_to_box_dist == float('inf'):
                 # Robot cannot reach the box
                 return float('inf') # This state is likely unsolvable

            total_heuristic += robot_to_box_dist

        # The heuristic is the sum of (box_distance + robot_to_box_distance) for each box needing move.
        # This is non-admissible as robot movement is counted for each box independently.
        return total_heuristic

