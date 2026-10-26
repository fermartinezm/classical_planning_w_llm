from collections import deque, defaultdict
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting."""
    return fact[1:-1].split()

class sokoban14Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to push all boxes to their goal positions. For each box, it calculates the minimal number of pushes needed to reach its goal and the robot's minimal distance to a position where it can start pushing the box. The sum of these values for all boxes gives the heuristic estimate.

    # Assumptions
    - The adjacency graph is undirected and fully connected (all locations are reachable).
    - Each box's goal is a single location specified in the problem's goals.
    - The robot can move freely between adjacent locations when not pushing boxes.

    # Heuristic Initialization
    - Parse static adjacency facts to build an adjacency list of the Sokoban grid.
    - Precompute the shortest path between all pairs of locations using BFS.
    - Extract goal locations for each box from the problem's goals.

    # Step-By-Step Thinking for Computing Heuristic
    1. For each box not at its goal:
        a. Compute the shortest path (number of pushes) from the box's current location to its goal.
        b. Compute the robot's shortest path to any adjacent location of the box's current position (to start pushing).
        c. Add the sum of these two values to the total heuristic.
    2. Sum the values for all boxes to get the heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic with static information and precompute shortest paths."""
        self.adjacency = defaultdict(list)
        self.goal_locations = {}
        self.shortest_paths = {}

        # Build adjacency list from static facts
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'adjacent':
                from_loc, to_loc = parts[1], parts[2]
                self.adjacency[from_loc].append(to_loc)

        # Precompute all-pairs shortest paths using BFS
        for loc in self.adjacency:
            distances = {loc: 0}
            queue = deque([loc])
            while queue:
                current = queue.popleft()
                for neighbor in self.adjacency[current]:
                    if neighbor not in distances:
                        distances[neighbor] = distances[current] + 1
                        queue.append(neighbor)
            self.shortest_paths[loc] = distances

        # Extract goal locations for each box
        for goal in task.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and parts[1].startswith('box'):
                self.goal_locations[parts[1]] = parts[2]

    def __call__(self, node):
        """Compute the heuristic value for the given state."""
        state = node.state
        total = 0

        # Extract robot's current location
        robot_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
                break
        if not robot_loc:
            return 0  # Invalid state if no robot location

        # Extract current box locations
        box_locs = {}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1].startswith('box'):
                box_locs[parts[1]] = parts[2]

        # Calculate heuristic for each box
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locs.get(box)
            if not current_loc or current_loc == goal_loc:
                continue  # Skip if box is at goal or not present

            # Get minimal pushes from current to goal
            pushes = self.shortest_paths.get(current_loc, {}).get(goal_loc, float('inf'))
            if pushes == float('inf'):
                continue  # Assume state is solvable, skip if path not found

            # Get robot's minimal distance to adjacent locations of the box
            adjacent_locs = self.adjacency.get(current_loc, [])
            if not adjacent_locs:
                continue  # No adjacent locations (invalid scenario)
            robot_distances = [self.shortest_paths.get(robot_loc, {}).get(adj, float('inf')) for adj in adjacent_locs]
            min_robot_dist = min(robot_distances)
            if min_robot_dist == float('inf'):
                continue  # Assume state is solvable, skip if no path

            total += pushes + min_robot_dist

        return total
