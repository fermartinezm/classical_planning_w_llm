# from heuristics.heuristic_base import Heuristic # Assuming this is provided externally
from collections import defaultdict, deque
from fnmatch import fnmatch

# Helper functions
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


# Inherit from Heuristic if available in the target environment
# class sokobanHeuristic(Heuristic):
class sokobanHeuristic:
    """
    A domain-dependent heuristic for the Sokoban domain.

    Estimates the cost to reach the goal by summing, for each box not at its goal:
    1. The minimum number of pushes required for the box to reach its goal
       (shortest path distance on the location graph).
    2. The minimum number of moves required for the robot to reach a position
       from which it can perform the first push for that box.

    This heuristic is not admissible. It simplifies the problem by:
    - Summing costs for boxes independently.
    - Calculating robot cost only for the first push of each box.
    - Ignoring complex deadlocks where boxes block each other or the robot.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting:
        - Goal locations for each box.
        - The location graph from adjacent facts.
        - Mapping from location pairs and direction to the opposite direction location.
        """
        self.goals = task.goals
        static_facts = task.static

        # Build the location graph from adjacent facts
        self.location_graph = defaultdict(list)
        # Store mapping from (loc1, loc2) -> dir for easier lookup
        self.adj_dirs = {}
        # Store mapping from (loc1, dir) -> loc2 for easier lookup
        self.adj_locs = {}
        # Store mapping from direction string to the opposite direction string
        self.opposite_dir = {'up': 'down', 'down': 'up', 'left': 'right', 'right': 'left'}

        all_locations = set()

        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                self.location_graph[loc1].append(loc2)
                self.adj_dirs[(loc1, loc2)] = direction
                self.adj_locs[(loc1, direction)] = loc2
                all_locations.add(loc1)
                all_locations.add(loc2)

        self.all_locations = frozenset(all_locations)

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def _shortest_path_info(self, start_loc, end_loc, obstacles=None):
        """
        Finds the shortest path distance and the first step location
        on a shortest path between two locations using BFS.
        Obstacles are locations that cannot be entered.
        Returns (distance, first_step_location) or (float('inf'), None).
        """
        if obstacles is None:
            obstacles = set()

        if start_loc not in self.all_locations or end_loc not in self.all_locations:
             return float('inf'), None

        if start_loc == end_loc:
            return 0, None # Distance 0, no step needed

        if start_loc in obstacles:
             return float('inf'), None

        queue = deque([(start_loc, 0, None)]) # (current_loc, dist, first_step_from_start)
        visited = {start_loc}

        while queue:
            current_loc, dist, initial_first_step = queue.popleft()

            if current_loc == end_loc:
                return dist, initial_first_step

            for neighbor in self.location_graph.get(current_loc, []):
                if neighbor not in visited and neighbor not in obstacles:
                    visited.add(neighbor)
                    # The first step from the start_loc is the neighbor visited directly from start_loc
                    step = neighbor if current_loc == start_loc else initial_first_step
                    queue.append((neighbor, dist + 1, step))

        return float('inf'), None # No path found

    def _get_push_position(self, box_loc, next_box_loc):
        """
        Given the current box location (?bloc) and the next location (?floc)
        after a push, determine the required robot location (?rloc) according
        to the PDDL definition: (adjacent ?rloc ?bloc ?dir) and (adjacent ?bloc ?floc ?dir).
        This means ?dir is the direction from robot to box, and from box to target.
        So, robot is behind the box relative to the push direction.

        Returns the required robot location string, or None if invalid.
        """
        # Find the direction from box_loc to next_box_loc
        push_dir = self.adj_dirs.get((box_loc, next_box_loc))
        if not push_dir:
            # next_box_loc is not adjacent to box_loc
            return None

        # We need ?rloc such that (adjacent ?rloc box_loc push_dir)
        # This means box_loc is adjacent to ?rloc in direction push_dir.
        # This is equivalent to finding the location adjacent to box_loc in the opposite direction of push_dir.
        opposite_dir_str = self.opposite_dir.get(push_dir)
        if not opposite_dir_str:
             return None # Should not happen for standard directions

        required_robot_loc = self.adj_locs.get((box_loc, opposite_dir_str))
        return required_robot_loc


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Get current robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None:
             # Robot location not found, state is invalid or malformed
             return float('inf')

        # Get current box locations
        current_box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name, loc = get_parts(fact)
                if obj_name in self.goal_locations: # Check if it's a box we care about
                    current_box_locations[obj_name] = loc

        total_heuristic = 0
        # Locations occupied by boxes are obstacles for the robot's movement
        robot_obstacles = set(current_box_locations.values())

        for box_name, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box_name)

            if current_loc is None:
                 # Box location not found, state is invalid or malformed
                 return float('inf')

            if current_loc == goal_loc:
                continue # Box is already at its goal

            # Component 1: Minimum pushes for the box
            # Calculate shortest path for the box from current_loc to goal_loc.
            # For the box's path distance calculation, we ignore temporary obstacles
            # like the robot or other boxes. Only static walls (locations not in graph) matter.
            box_path_dist, first_step_loc = self._shortest_path_info(current_loc, goal_loc, obstacles=None) # No obstacles for box path distance

            if box_path_dist == float('inf'):
                 # Box cannot reach the goal location at all (e.g., trapped by walls)
                 # This is a static deadlock. Assign infinity.
                 return float('inf')

            # Component 2: Robot positioning cost for the first push
            # Find the required robot location to push the box from current_loc to first_step_loc
            # This requires first_step_loc to be a valid next step (box_path_dist > 0)
            if box_path_dist > 0 and first_step_loc is not None:
                 required_robot_loc = self._get_push_position(current_loc, first_step_loc)

                 if required_robot_loc is None:
                      # This implies first_step_loc is not a valid adjacent location or direction mapping is wrong.
                      # Should not happen if first_step_loc came from a valid shortest path.
                      # Treat as unreachable.
                      return float('inf')

                 # Calculate shortest path for the robot to reach required_robot_loc.
                 # Obstacles for the robot are locations occupied by any box.
                 robot_path_dist, _ = self._shortest_path_info(robot_loc, required_robot_loc, obstacles=robot_obstacles)

                 if robot_path_dist == float('inf'):
                      # Robot cannot reach the required push position (e.g., trapped by boxes/walls)
                      # This is a dynamic or static deadlock for the robot. Assign infinity.
                      return float('inf')

                 # Heuristic contribution for this box: pushes + robot moves to first push position
                 total_heuristic += box_path_dist + robot_path_dist
            elif box_path_dist > 0 and first_step_loc is None:
                 # This case should ideally not be reached if box_path_dist is finite > 0,
                 # as _shortest_path_info should return a first step.
                 # If it happens, it indicates an unexpected state or graph issue.
                 # Treat as unreachable.
                 return float('inf')
            # else: box_path_dist is 0, box is at goal, loop continues.

        # The total_heuristic is 0 if and only if all boxes were at their goals
        # and no infinity was returned. This correctly represents the goal state.
        # For any non-goal state that is reachable, total_heuristic will be > 0.
        # For unreachable states, it returns infinity.

        return total_heuristic
