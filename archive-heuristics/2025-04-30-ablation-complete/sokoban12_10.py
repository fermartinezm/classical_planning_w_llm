from fnmatch import fnmatch
from collections import deque
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Dummy Heuristic base class if not provided externally
class Heuristic:
    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

    def __call__(self, node):
        raise NotImplementedError

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

def bfs(start_node, graph):
    """
    Performs BFS from start_node on the graph.
    Returns a dictionary mapping reachable nodes to their distance from start_node.
    """
    distances = {start_node: 0}
    queue = deque([start_node])
    visited = {start_node}

    while queue:
        current_node = queue.popleft()

        # Ensure current_node exists in the graph keys before accessing neighbors
        if current_node in graph:
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the required number of actions to reach a goal state
    by summing the estimated costs for each box that is not yet at its goal location.
    The estimated cost for a single box includes the minimum number of pushes required
    to move the box to its goal location (shortest path distance on the location graph)
    plus the minimum number of robot movements required to get into position for the
    first push of that box. Subsequent robot movements between pushes for the same
    box are approximated as zero cost, assuming a straight-line push path where the
    robot ends up in the correct position for the next push.

    # Assumptions
    - The locations form a graph defined by `adjacent` predicates.
    - Distances between locations are shortest path distances on this graph.
    - The cost of a `move` action is 1.
    - The cost of a `push` action is 1.
    - The heuristic ignores potential deadlocks (situations where a box cannot be moved further towards the goal).
    - The heuristic ignores conflicts between multiple boxes or obstacles (walls not defined by the graph structure) that might block paths.
    - The heuristic assumes the robot can always reach the required push position if it exists in the graph.

    # Heuristic Initialization
    - Parse `adjacent` facts to build the location graph (`adj`).
    - Parse `adjacent` facts to build the `push_pos` map, which stores the required robot location `rloc` for pushing a box from `bloc` to `floc`.
    - Compute all-pairs shortest paths between all locations using BFS on the location graph and store them in `dist`.
    - Parse goal conditions to store the target location for each box in `goal_locations`.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot (`robot_loc`).
    2. Identify the current location of each box (`box_locs`).
    3. Initialize the total heuristic cost `total_cost` to 0.
    4. For each box `b` and its goal location `goal_b` from `goal_locations`:
        a. Get the box's current location `loc_b = box_locs.get(box)`. If the box is not found in the state, return infinity.
        b. If `loc_b` is the same as `goal_b`, this box is already in place; continue to the next box.
        c. Calculate the shortest path distance for the box from `loc_b` to `goal_b` using the precomputed distances (`dist[loc_b][goal_b]`). If the goal is unreachable for the box, return infinity. This distance represents the minimum number of `push` actions required for this box. Add this distance to `total_cost`.
        d. Find a location `l_1` that is the first step on a shortest path for the box from `loc_b` to `goal_b`. This is a neighbor `l_1` of `loc_b` such that `dist[l_1][goal_b]` is exactly one less than `dist[loc_b][goal_b]`. If no such neighbor exists (e.g., goal is unreachable or a complex path is needed), return infinity.
        e. Determine the required robot location `r_0` to push the box from `loc_b` to `l_1`. This is found using the precomputed `push_pos[(loc_b, l_1)]`. If this push configuration is not possible according to static facts, return infinity.
        f. Calculate the shortest path distance for the robot from its current location `robot_loc` to the required push position `r_0` using the precomputed distances (`dist[robot_loc][r_0]`). If the required robot position is unreachable, return infinity. Add this distance to `total_cost`.
    5. Return the `total_cost`. If any required distance calculation or lookup failed, infinity would have been returned earlier.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        self.static = task.static

        # Data structures to build from static facts
        self.locations = set()
        self.adj = {} # Graph: loc -> {neighbor: direction}
        self.push_pos = {} # Map: (bloc, floc) -> rloc (required robot pos)

        # 1. Build the location graph and push_pos map
        adjacent_facts = [fact for fact in self.static if match(fact, "adjacent", "*", "*", "*")]
        for fact in adjacent_facts:
            _, loc1, loc2, direction = get_parts(fact)
            self.locations.add(loc1)
            self.locations.add(loc2)

            if loc1 not in self.adj:
                self.adj[loc1] = {}
            self.adj[loc1][loc2] = direction

        # Build push_pos map: (bloc, floc) -> rloc
        # Iterate through all triplets of locations rloc, bloc, floc such that
        # adjacent(rloc, bloc, dir) and adjacent(bloc, floc, dir) for some direction dir
        for fact1 in adjacent_facts:
            _, rloc, bloc, dir1 = get_parts(fact1)
            for fact2 in adjacent_facts:
                _, bloc2, floc, dir2 = get_parts(fact2)
                # Check if fact2 starts where fact1 ends, and directions match
                if bloc == bloc2 and dir1 == dir2:
                    self.push_pos[(bloc, floc)] = rloc

        # 2. Compute all-pairs shortest paths
        self.dist = {}
        for start_loc in self.locations:
            self.dist[start_loc] = bfs(start_loc, self.adj)

        # 3. Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached (heuristic is 0)
        if self.goals <= state:
             return 0

        # Identify current robot and box locations
        robot_loc = None
        box_locs = {} # box -> location
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at" and parts[1] in self.goal_locations: # Only track boxes that are goals
                 box_locs[parts[1]] = parts[2]

        # Ensure robot location is found
        if robot_loc is None:
             # This should not happen in a valid Sokoban state, but handle defensively
             return float('inf') # Cannot solve without robot

        total_cost = 0

        # Calculate cost for each box not at its goal
        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locs.get(box) # Get current location

            if current_box_loc is None:
                 # Box is not 'at' any location predicate in the state - problem state?
                 return float('inf')

            if current_box_loc == goal_loc:
                continue # Box is already at goal

            # Cost 1: Minimum pushes for the box
            # Check if current_box_loc is a valid start node for distance lookup
            if current_box_loc not in self.dist:
                 return float('inf') # Current box location is not in the graph

            # Check if goal_loc is reachable from current_box_loc
            if goal_loc not in self.dist[current_box_loc]:
                 return float('inf') # Goal is unreachable for the box

            box_push_dist = self.dist[current_box_loc][goal_loc]
            total_cost += box_push_dist

            # Cost 2: Robot movement to get into position for the first push
            # Find the first step location l_1 on a shortest path from current_box_loc to goal_loc
            l_1 = None
            # We need a neighbor of current_box_loc that is one step closer to goal_loc
            if current_box_loc in self.adj: # Ensure current_box_loc has neighbors in the graph
                for neighbor in self.adj[current_box_loc]:
                    # Check if neighbor is reachable from current_box_loc (always true if in adj)
                    # and goal is reachable from neighbor
                    if goal_loc in self.dist.get(neighbor, {}): # Use .get for safety
                         # Check if neighbor is one step closer to the goal
                         if self.dist[neighbor][goal_loc] == box_push_dist - 1:
                             l_1 = neighbor
                             break # Found a suitable first step

            # If box_push_dist > 0, we must find a valid first step l_1
            if box_push_dist > 0 and l_1 is None:
                 # Cannot find a neighbor that reduces distance to goal.
                 # This might indicate a non-standard path or an issue, treat as unsolvable by this heuristic.
                 return float('inf')
            elif box_push_dist == 0:
                 # Box is at goal, but we already handled this case. This branch should not be reached.
                 # If reached, something is inconsistent.
                 if current_box_loc != goal_loc: # Should be equal if dist is 0
                     return float('inf')
                 else: # Box is at goal, cost is 0, already handled by continue
                     pass # Should not add robot cost if box is at goal

            # If box_push_dist > 0, l_1 is found. Find the required robot position r_0.
            if box_push_dist > 0:
                # Find the required robot position r_0 to push from current_box_loc to l_1
                # The key for push_pos is (bloc, floc)
                push_key = (current_box_loc, l_1)
                if push_key not in self.push_pos:
                     # Cannot push the box from current_box_loc towards l_1 based on static facts.
                     # This configuration might be impossible or require complex setup.
                     return float('inf')

                required_robot_loc = self.push_pos[push_key]

                # Calculate robot distance to the required push position
                # Check if robot_loc is a valid start node for distance lookup
                if robot_loc not in self.dist:
                     return float('inf') # Robot location is not in the graph

                # Check if required_robot_loc is reachable from robot_loc
                if required_robot_loc not in self.dist[robot_loc]:
                     return float('inf') # Required robot position is unreachable

                robot_move_dist = self.dist[robot_loc][required_robot_loc]
                total_cost += robot_move_dist

        return total_cost
