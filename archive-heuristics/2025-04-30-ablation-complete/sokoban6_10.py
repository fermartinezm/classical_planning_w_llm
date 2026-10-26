import re
from collections import deque
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions for parsing PDDL facts
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

# Sokoban specific helper functions
def parse_location(loc_str):
    """Parses a location string like 'loc_R_C' into a tuple (R, C)."""
    match = re.match(r'loc_(\d+)_(\d+)', loc_str)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None # Should not happen in valid sokoban instances

def get_opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == 'up': return 'down'
    if direction == 'down': return 'up'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen

def build_grid_graph(static_facts):
    """
    Builds an adjacency list representation of the grid graph from adjacent facts.
    Also returns a set of all valid locations.
    Graph: {location_str: {neighbor_location_str: direction_str}}
    """
    graph = {}
    locations = set()
    for fact in static_facts:
        if match(fact, "adjacent", "*", "*", "*"):
            _, loc1, loc2, direction = get_parts(fact)
            locations.add(loc1)
            locations.add(loc2)
            if loc1 not in graph:
                graph[loc1] = {}
            graph[loc1][loc2] = direction
            # Add the reverse edge as well, with opposite direction
            if loc2 not in graph:
                graph[loc2] = {}
            graph[loc2][loc1] = get_opposite_direction(direction)
    return graph, locations

def precompute_shortest_paths(graph, locations):
    """
    Computes all-pairs shortest paths using BFS.
    Returns a dictionary dist[loc1][loc2] = distance.
    Returns float('inf') if unreachable.
    """
    dist = {}
    for start_node in locations:
        dist[start_node] = {}
        for loc in locations:
             dist[start_node][loc] = float('inf')

        queue = deque([(start_node, 0)])
        dist[start_node][start_node] = 0
        visited = {start_node}

        while queue:
            current_node, current_dist = queue.popleft()

            if current_node in graph: # Ensure the node exists in the graph keys
                for neighbor in graph[current_node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist[start_node][neighbor] = current_dist + 1
                        queue.append((neighbor, current_dist + 1))
    return dist


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions required to move all boxes
    to their goal locations. It sums the estimated cost for each box independently.
    The estimated cost for a single box is the sum of:
    1. The shortest path distance for the robot to reach a location adjacent
       to the box from which it can push the box towards its goal.
    2. The shortest path distance for the box to reach its goal location
       (which is the minimum number of pushes required).

    # Assumptions
    - The grid structure is defined by the 'adjacent' predicates.
    - Shortest paths on this grid can be precomputed.
    - The cost of moving the robot and pushing the box are both 1.
    - The heuristic ignores potential conflicts between boxes or complex deadlocks,
      except for simple structural deadlocks where a box cannot be pushed towards
      its goal from any valid adjacent robot position.
    - The goal only specifies the final locations of boxes, not the robot.

    # Heuristic Initialization
    - Parses goal conditions to map each box to its goal location.
    - Parses static facts ('adjacent' predicates) to build the grid graph.
    - Precomputes all-pairs shortest path distances on the grid graph.
    - Stores the graph and distances for use in the heuristic calculation.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box.
    2. Initialize the total heuristic cost to 0.
    3. For each box specified in the goal conditions:
       a. If the box is already at its goal location, its contribution is 0. Continue to the next box.
       b. Get the box's current location (`L_box`) and its goal location (`L_goal`).
       c. Calculate the shortest path distance from `L_box` to `L_goal` using the precomputed distances (`box_dist`). If the goal is unreachable for the box (`box_dist` is infinity), the problem is likely unsolvable from this state; return infinity for the total heuristic.
       d. Find potential next locations (`L_next`) for the box: these are neighbors of `L_box` that are one step closer to `L_goal` according to the precomputed distances (`dist[L_next][L_goal] == box_dist - 1`).
       e. Filter potential `L_next` locations: For each `L_next`, determine the required push direction (`dir`) from `L_box` to `L_next`. Find the required robot push location (`L_robot_push`) adjacent to `L_box` in the opposite direction (`opposite_direction(dir)`). Keep only those `L_next` for which `L_robot_push` exists in the grid graph.
       f. If no valid `L_next` exists after filtering (meaning the box cannot be pushed towards the goal from any valid robot position towards the goal), the box is structurally deadlocked; return infinity for the total heuristic.
       g. Choose one valid candidate `L_robot_push` from the filtered list that minimizes the robot's travel distance from its current location.
       h. Calculate the shortest path distance from the robot's current location (`L_robot_current`) to the chosen `L_robot_push` location using precomputed distances (`robot_dist`). If the push location is unreachable for the robot (`robot_dist` is infinity), return infinity for the total heuristic.
       i. The heuristic contribution for this box is `robot_dist + box_dist`. Add this to the total heuristic.
    4. Return the total heuristic cost. If any step resulted in infinity, return infinity.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                package, location = args
                self.goal_locations[package] = location

        # Build the grid graph and get all valid locations from static facts.
        self.graph, self.locations = build_grid_graph(static_facts)

        # Precompute all-pairs shortest path distances on the grid graph.
        self.dist = precompute_shortest_paths(self.graph, self.locations)

        # Define infinity for unreachable states
        self.infinity = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current robot location
        robot_location = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_location = get_parts(fact)[1]
                break
        if robot_location is None:
             # Robot location not found, should not happen in valid states
             # This indicates an invalid state representation
             return self.infinity

        # Find current box locations
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # Only consider objects that are boxes and are in the goal
                 if obj in self.goal_locations:
                    box_locations[obj] = loc

        total_heuristic = 0

        for box, goal_location in self.goal_locations.items():
            # Ensure the box is currently located somewhere in the state
            if box not in box_locations:
                 # This box is not in the state's 'at' facts.
                 # This shouldn't happen in a valid Sokoban state representation.
                 # Return infinity as it's likely an invalid or unsolvable state.
                 return self.infinity

            current_box_location = box_locations[box]

            # 3a. If the box is already at its goal location, its contribution is 0.
            if current_box_location == goal_location:
                continue

            # Ensure current box location and goal location are in the graph
            if current_box_location not in self.locations or goal_location not in self.locations:
                 # Box or goal location is not part of the defined grid
                 return self.infinity # Invalid state or unsolvable

            # 3c. Calculate box_dist
            box_dist = self.dist[current_box_location][goal_location]
            if box_dist == self.infinity:
                # Box cannot reach the goal location on the static grid
                return self.infinity # Problem is unsolvable

            # 3d, 3e. Find valid candidate next locations and required robot push locations
            # We are looking for a neighbor L_next of current_box_location
            # such that dist(L_next, goal_location) == box_dist - 1
            # AND the robot push location (neighbor of current_box_location in opposite direction) exists.
            valid_robot_push_locations = set()

            if current_box_location in self.graph: # Ensure current location is in graph
                for L_next, dir_to_next in self.graph[current_box_location].items():
                    # Check if L_next is one step closer to the goal
                    if self.dist[L_next][goal_location] == box_dist - 1:
                        # Determine required robot push location
                        required_robot_dir = get_opposite_direction(dir_to_next)
                        L_robot_push = None
                        # Find the neighbor of current_box_location in the required_robot_dir
                        if current_box_location in self.graph: # Redundant check, but safe
                            for neighbor_loc, neighbor_dir in self.graph[current_box_location].items():
                                if neighbor_dir == required_robot_dir:
                                    L_robot_push = neighbor_loc
                                    break

                        # 3e. Filter candidates: Check if L_robot_push exists in the graph
                        if L_robot_push is not None and L_robot_push in self.locations:
                             valid_robot_push_locations.add(L_robot_push)

            # 3f. If no valid robot push location exists for any step towards the goal, the box is deadlocked
            if not valid_robot_push_locations:
                 # Box cannot be pushed towards the goal from any valid adjacent robot position
                 return self.infinity # Box is deadlocked

            # 3g, 3h. Choose the valid robot push location that minimizes robot distance
            best_robot_dist = self.infinity

            # Ensure robot_location is in the graph before looking up distances
            if robot_location not in self.locations:
                 return self.infinity # Robot is in an invalid location

            for L_robot_push_candidate in valid_robot_push_locations:
                 # Ensure the candidate push location is in the graph
                 if L_robot_push_candidate in self.locations:
                     robot_dist_candidate = self.dist[robot_location][L_robot_push_candidate]
                     best_robot_dist = min(best_robot_dist, robot_dist_candidate)

            # If the best robot distance is still infinity, the robot cannot reach any valid push location
            if best_robot_dist == self.infinity:
                 return self.infinity # Robot cannot reach any push position for this box

            # 3i. Add contribution: robot_dist + box_dist
            total_heuristic += best_robot_dist + box_dist

        # 4. Return the total heuristic cost.
        return total_heuristic
