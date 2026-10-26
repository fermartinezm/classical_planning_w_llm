from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# If not, you might need to define a simple base class or remove inheritance.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    # Define a dummy base class if the actual one is not available
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        # Handle unexpected fact format, though planner typically provides strings
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_3_5)".
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
    This heuristic estimates the total effort required to move all boxes to their
    respective goal locations. It sums the shortest path distance for each box
    to its goal and the shortest path distance for the robot to reach that box.

    # Assumptions
    - Each box has a unique, fixed goal location specified in the problem's goal state.
    - The grid structure and connectivity are defined by the `adjacent` static facts.
    - The heuristic assumes a simple sum of independent movements (box movement towards goal
      and robot movement towards the box), ignoring complex interactions like
      blocking or needing to move boxes away from goals temporarily.

    # Heuristic Initialization
    - Extracts the goal location for each box from the task's goal conditions.
    - Builds a graph representation of the locations based on `adjacent` static facts.
    - Precomputes the shortest path distance between all pairs of locations using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Identify the robot's current location from the state.
    2.  Identify the current location for each box from the state.
    3.  Initialize the total heuristic cost to 0.
    4.  For each box defined in the goal conditions:
        a.  Get the box's current location and its target goal location.
        b.  If the box is already at its goal location, it contributes 0 to the heuristic.
        c.  If the box is not at its goal location:
            i.  Calculate the shortest path distance from the box's current location
                to its goal location using the precomputed distances. This estimates
                the minimum number of push actions required for the box itself,
                assuming a clear path and the robot being in position.
            ii. Calculate the shortest path distance from the robot's current location
                to the box's current location using the precomputed distances. This
                estimates the minimum number of move actions required for the robot
                to reach the box.
            iii. Add the sum of these two distances (box-to-goal distance + robot-to-box distance)
                 to the total heuristic cost.
    5.  Return the total calculated cost. If any required distance is infinite (locations
        are disconnected), return infinity, indicating an likely unsolvable state.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building the location graph."""
        super().__init__(task) # Call base class constructor if it exists
        self.goals = task.goals
        static_facts = task.static

        # Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            # Assuming goal facts are (at box_name goal_location)
            if predicate == "at" and len(args) == 2 and args[0].startswith("box"):
                box, location = args
                self.goal_locations[box] = location

        # Build the location graph from adjacent facts
        self.graph = {}
        all_locations = set()
        for fact in static_facts:
            # Assuming adjacent facts are (adjacent loc1 loc2 direction)
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, _ = get_parts(fact)
                all_locations.add(loc1)
                all_locations.add(loc2)
                # Adjacent is bidirectional in Sokoban grid
                self.graph.setdefault(loc1, set()).add(loc2)
                self.graph.setdefault(loc2, set()).add(loc1)

        # Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in all_locations:
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0
            while q:
                current_loc, dist = q.popleft()
                # Ensure current_loc is a valid key in the graph
                if current_loc in self.graph:
                    for neighbor in self.graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_loc, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

    def __call__(self, node):
        """Estimate the minimum cost to move all boxes to their goal locations."""
        state = node.state

        # Find robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        # If robot location isn't found, the state is invalid or unsolvable
        if robot_loc is None:
             return float('inf')

        # Find current box locations
        box_locations = {}
        for fact in state:
            # Assuming box facts are (at box_name location)
            if match(fact, "at", "box*", "*"):
                 box, loc = get_parts(fact)[1:]
                 box_locations[box] = loc

        total_cost = 0

        # Sum distances for each box not at its goal
        for box_name, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box_name)

            # If a box from the goal is not found in the current state, something is wrong
            if current_box_loc is None:
                 return float('inf')

            if current_box_loc != goal_loc:
                # Get distance from box to its goal
                # Use .get with default float('inf') to handle cases where goal is unreachable
                dist_box_goal = self.distances.get((current_box_loc, goal_loc), float('inf'))
                if dist_box_goal == float('inf'):
                    # Goal is unreachable from the box's current location
                    return float('inf')

                # Get distance from robot to the box
                dist_robot_box = self.distances.get((robot_loc, current_box_loc), float('inf'))
                # If robot cannot reach the box, this state is likely unsolvable or requires complex maneuvers
                if dist_robot_box == float('inf'):
                     return float('inf')

                # Add the combined distance to the total cost
                total_cost += dist_box_goal + dist_robot_box

        return total_cost

