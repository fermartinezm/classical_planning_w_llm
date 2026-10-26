from fnmatch import fnmatch
# Assuming Heuristic base class is available as in the examples
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if running standalone for syntax check
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            pass


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or invalid format gracefully
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
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

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost as the sum of shortest path distances from each
    misplaced box to its goal, plus the shortest path distance from the
    robot to the closest misplaced box.

    Shortest paths are calculated using BFS on the grid graph defined by
    adjacent facts, considering other boxes and non-clear locations as obstacles.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by building the location graph and
        mapping boxes to their goal locations.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Build the graph from adjacent facts
        self.graph = {} # Adjacency list: loc -> set of adjacent locs

        for fact in self.static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                l1, l2, dir = get_parts(fact)[1:]
                # Add edge representing undirected connectivity
                # The PDDL example lists both directions explicitly, but adding
                # the reverse here ensures the graph is undirected based on
                # any adjacency relationship existing.
                self.graph.setdefault(l1, set()).add(l2)
                self.graph.setdefault(l2, set()).add(l1)


        # Map boxes to their goal locations
        self.goal_locations = {} # box name -> goal location string
        for goal in self.goals:
            if match(goal, "at", "*", "*"):
                box, loc = get_parts(goal)[1:]
                self.goal_locations[box] = loc

    def bfs(self, start, end, graph, state, box_locations, moving_object):
        """
        Find the shortest path distance from start to end in the graph,
        considering dynamic obstacles based on the current state.

        Args:
            start (str): The starting location.
            end (str): The target location.
            graph (dict): The adjacency list representation of the grid.
            state (frozenset): The current state facts.
            box_locations (dict): Mapping from box name to its current location.
            moving_object (str): 'robot' or the name of the box being moved.

        Returns:
            int: The shortest distance, or float('inf') if unreachable.
        """
        if start == end:
            return 0

        # Ensure start and end locations are valid nodes in the graph
        if start not in graph or end not in graph:
             return float('inf') # Cannot reach if locations are not in the grid

        # Helper functions to check if a location is an obstacle
        def is_clear(loc):
            return f"(clear {loc})" in state

        def is_occupied_by_box(loc):
            return loc in box_locations.values()

        def is_occupied_by_other_box(loc, current_box):
             return loc in {box_locations[b] for b in box_locations if b != current_box}

        queue = [(start, 0)]
        visited = {start}

        while queue:
            current, dist = queue.pop(0)

            if current == end:
                return dist

            # Ensure current location is in the graph (should be if from state/goals/adjacent)
            if current not in graph:
                 continue # Should not happen if graph is built correctly

            for neighbor in graph[current]:
                if neighbor not in visited:
                    is_obstacle = False

                    # A location is an obstacle if it's not clear
                    if not is_clear(neighbor):
                        is_obstacle = True
                    # A location is an obstacle if it's occupied by a box (for robot)
                    elif moving_object == 'robot' and is_occupied_by_box(neighbor):
                        is_obstacle = True
                    # A location is an obstacle if it's occupied by another box (for a box)
                    elif moving_object in self.goal_locations and is_occupied_by_other_box(neighbor, moving_object):
                         is_obstacle = True

                    # A box can only be pushed into a location that is clear.
                    # The robot can only move into a location that is clear.
                    # The BFS finds a path through *currently* clear and unoccupied locations.

                    if not is_obstacle:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return float('inf') # End not reachable

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # Find robot and box locations in the current state
        robot_location = None
        box_locations = {} # box name -> current location string

        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
            elif match(fact, "at", "*", "*"):
                box, loc = get_parts(fact)[1:]
                box_locations[box] = loc

        # Identify boxes that are not at their goal locations
        misplaced_boxes = {
            box for box, loc in box_locations.items()
            if box in self.goal_locations and loc != self.goal_locations[box]
        }

        # If all relevant boxes are at their goals, the heuristic is 0
        if not misplaced_boxes:
            return 0

        # Calculate sum of box-goal distances
        box_goal_dist_sum = 0
        for box in misplaced_boxes:
            current_loc = box_locations[box]
            goal_loc = self.goal_locations[box]

            # Calculate shortest path for the box to its goal
            # Obstacles for box movement: locations that are not clear or occupied by other boxes
            dist = self.bfs(current_loc, goal_loc, self.graph, state, box_locations, box)

            if dist == float('inf'):
                # If any box cannot reach its goal, the state is likely a deadlock
                return float('inf') # Return infinity immediately if any box is stuck

            box_goal_dist_sum += dist

        # Calculate minimum robot distance to any misplaced box
        min_robot_box_dist = float('inf')
        for box in misplaced_boxes:
            box_loc = box_locations[box]
            # Calculate shortest path for the robot to the box's location
            # Obstacles for robot movement: locations that are not clear or occupied by any box
            dist = self.bfs(robot_location, box_loc, self.graph, state, box_locations, 'robot')

            # Use the distance to the box location as the robot cost proxy.
            # If the robot cannot reach the box location, this box is unreachable by the robot.
            if dist != float('inf'):
                 min_robot_box_dist = min(min_robot_box_dist, dist)

        # If the robot cannot reach any misplaced box, the state is likely a deadlock
        if min_robot_box_dist == float('inf') and misplaced_boxes:
             return float('inf') # Return infinity if robot cannot reach any box

        # Total heuristic is sum of box-goal distances + minimum robot-box distance
        # Note: If min_robot_box_dist was initialized to inf and no reachable boxes were found,
        # the sum will be inf + inf = inf, which is correct for a deadlock.
        return box_goal_dist_sum + min_robot_box_dist
