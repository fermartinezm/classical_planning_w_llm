from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Check if the number of parts matches the number of args
    if len(parts) != len(args): return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach a goal state by summing the shortest
    path distances for each misplaced box to its corresponding goal location.
    The distance is calculated on the static grid graph defined by 'adjacent'
    predicates, ignoring dynamic obstacles like other boxes or the robot.

    # Assumptions
    - The problem instance defines a grid-like structure via 'adjacent' predicates.
    - Each box that needs to be moved has a unique goal location specified in the goal state.
    - The distance calculation for a box to its goal ignores the robot's position
      and the positions of other boxes or non-'clear' locations in the current state.
      It only considers the static grid connectivity.
    - Adjacency is symmetric (if A is adjacent to B, B is adjacent to A).
    - The `task.goals` attribute provides a list of simple goal fact strings (e.g., `['(at box1 loc1)', '(at box2 loc2)']`).

    # Heuristic Initialization
    - Parses the goal conditions (`task.goals`) to identify the target location for each box.
    - Parses the static facts (`task.static`) to build an adjacency list representation of the
      grid graph based on 'adjacent' predicates.

    # Step-By-Step Thinking for Computing Heuristic
    1. Identify the current location of each box that has a goal location specified by iterating
       through the facts in the current state.
    2. Initialize the total heuristic value to 0.
    3. For each box that has a goal location:
        a. Retrieve the box's current location from the state and its goal location from the
           initialized goal data.
        b. If the box's current location is different from its goal location:
            i. Calculate the shortest path distance between the current location and
               the goal location using Breadth-First Search (BFS) on the static
               grid graph constructed during initialization. This distance represents
               the minimum number of steps the box itself would need to move along
               the defined grid paths.
            ii. Add this calculated distance to the total heuristic value.
            iii. If the goal location is unreachable from the current location on the
                static graph (BFS returns infinity), the state is likely unsolvable
                or extremely costly; return a very large value (infinity) immediately.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal locations for boxes
        and building the adjacency graph from static facts.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        # Assuming task.goals is a list of goal fact strings, e.g., ['(at box1 loc1)', '(at box2 loc2)']
        for goal_fact_str in self.goals:
            parts = get_parts(goal_fact_str)
            # We only care about (at ?box ?location) goal facts with exactly 3 parts
            if parts and parts[0] == 'at' and len(parts) == 3:
                 box_name = parts[1]
                 goal_loc = parts[2]
                 self.goal_locations[box_name] = goal_loc

        # Build the adjacency list from static facts.
        self.adjacency_list = {}
        for fact in static_facts:
            parts = get_parts(fact)
            # Assuming adjacent facts have 4 parts: (adjacent loc1 loc2 dir)
            if parts and parts[0] == 'adjacent' and len(parts) == 4:
                loc1 = parts[1]
                loc2 = parts[2]
                # Assuming adjacency is symmetric, add both directions
                self.adjacency_list.setdefault(loc1, []).append(loc2)
                self.adjacency_list.setdefault(loc2, []).append(loc1)

    def bfs_distance(self, start, end, adj_list):
        """
        Calculates the shortest path distance between two locations
        using BFS on the adjacency graph. Returns float('inf') if unreachable.
        """
        if start == end:
            return 0

        # Handle cases where start or end might not be in the graph (e.g., isolated locations)
        if start not in adj_list or end not in adj_list:
             # If start or end is not a node in the connected graph, distance is infinite
             # unless start == end (handled above).
             return float('inf')

        queue = deque([(start, 0)]) # (location, distance)
        visited = {start}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end:
                return dist

            # Neighbors are guaranteed to be in adj_list keys if current_loc is,
            # because we add both directions during initialization.
            # Check if current_loc has neighbors defined
            if current_loc in adj_list:
                for neighbor in adj_list[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        # If BFS completes without finding the end location
        return float('inf')

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        (pushes) to move all misplaced boxes to their goal locations.
        """
        state = node.state

        # Find current locations of all boxes that have a goal.
        current_box_locations = {}
        # Iterate through state facts to find box locations
        for fact in state:
            parts = get_parts(fact)
            # Look for (at ?box ?location) facts
            if parts and parts[0] == 'at' and len(parts) == 3:
                 obj_name = parts[1]
                 current_loc = parts[2]
                 # Only store location if the object is a box we care about (i.e., has a goal)
                 if obj_name in self.goal_locations:
                    current_box_locations[obj_name] = current_loc

        total_heuristic = 0

        # Sum distances for all misplaced boxes.
        for box_name, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box_name)

            # Defensive check: A box with a goal should be 'at' some location in the state.
            # If not, the state might be malformed or represent something unexpected.
            # Returning infinity signals this state is likely problematic or very far.
            if current_loc is None:
                 # print(f"Warning: Box {box_name} with goal {goal_loc} not found in state.")
                 return float('inf') # State is missing a box with a goal

            if current_loc != goal_loc:
                dist = self.bfs_distance(current_loc, goal_loc, self.adjacency_list)

                # If a box goal is unreachable from its current location on the static graph
                if dist == float('inf'):
                     # print(f"Warning: Goal {goal_loc} unreachable for box {box_name} from {current_loc}.")
                     return float('inf') # This state is likely unsolvable

                total_heuristic += dist

        return total_heuristic
