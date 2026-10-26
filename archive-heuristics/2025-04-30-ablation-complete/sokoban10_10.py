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

def get_opposite_direction(direction):
    """Returns the opposite direction."""
    if direction == "up": return "down"
    if direction == "down": return "up"
    if direction == "left": return "right"
    if direction == "right": return "left"
    return None # Should not happen in this domain

def build_adjacency_graphs(static_facts):
    """
    Builds adjacency graphs from static facts.
    adj_graph: maps location -> { (neighbor_loc, direction) }
    adj_graph_reverse: maps target_loc -> { (source_loc, direction) }
    """
    adj_graph = {}
    adj_graph_reverse = {}
    locations = set()

    for fact in static_facts:
        if match(fact, "adjacent", "*", "*", "*"):
            _, l1, l2, direction = get_parts(fact)
            locations.add(l1)
            locations.add(l2)
            adj_graph.setdefault(l1, set()).add((l2, direction))
            adj_graph_reverse.setdefault(l2, set()).add((l1, direction))

    return adj_graph, adj_graph_reverse, list(locations)

def compute_distances(locations, adj_graph):
    """
    Computes all-pairs shortest paths using BFS from each location.
    Returns a dictionary distances[(l1, l2)] = shortest_path_distance.
    """
    distances = {}
    # Use a large value for unreachable nodes
    unreachable_distance = float('inf')

    for start_node in locations:
        q = deque([(start_node, 0)])
        visited = {start_node: 0}
        distances[(start_node, start_node)] = 0

        while q:
            current_node, dist = q.popleft()

            for neighbor, _ in adj_graph.get(current_node, []):
                if neighbor not in visited:
                    visited[neighbor] = dist + 1
                    distances[(start_node, neighbor)] = dist + 1
                    q.append((neighbor, dist + 1))

    return distances


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the number of actions needed to move all boxes
    to their goal locations. It sums the estimated cost for each box
    independently. The cost for a single box is estimated as the minimum
    number of pushes required for the box to reach its goal, plus the minimum
    number of robot moves required to get into position for the *first* push
    towards the goal, plus an estimated cost for robot repositioning between
    subsequent pushes.

    # Assumptions
    - The location names follow a grid-like structure (e.g., loc_row_col),
      although the heuristic uses the adjacency graph directly.
    - The graph defined by adjacent predicates is connected for relevant locations.
    - The heuristic does not consider deadlocks or the blocking effects of
      other boxes or walls on the robot's path or the box's path beyond the
      distance calculation on the full location graph.
    - The cost of a 'move' action is 1.
    - The cost of a 'push' action is 1.
    - Robot repositioning between pushes is estimated to cost 1 move per push
      after the first one. This is a simplification; the actual cost depends
      on the specific path and required push positions.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task goals.
    - Builds the adjacency graph and its reverse from static 'adjacent' facts.
    - Computes all-pairs shortest paths on the location graph using BFS.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the current location of the robot and each box.
    2. Initialize the total heuristic value to 0.
    3. For each box that is not currently at its specific goal location:
        a. Get the box's current location (`loc_b`) and its goal location (`goal_b`).
        b. Calculate the shortest path distance for the box from `loc_b` to `goal_b`
           on the location graph (`box_dist`). This represents the minimum number
           of push actions required for the box itself. If the goal is unreachable,
           the distance is infinite, and the heuristic returns infinity.
        c. Find a location `loc_next` adjacent to `loc_b` that is one step closer
           to `goal_b` along a shortest path (i.e., `dist(loc_next, goal_b) == box_dist - 1`).
           If multiple such locations exist, pick any one. If no such location exists
           (should only happen if box_dist is 0 or goal is unreachable), handle appropriately.
        d. Determine the direction (`dir`) from `loc_b` to `loc_next` using the
           adjacency graph.
        e. Determine the required robot location (`rloc`) to push the box from
           `loc_b` to `loc_next` in direction `dir`. Based on the PDDL action,
           the robot must be at `rloc` such that moving from `rloc` to `loc_b`
           is in direction `dir`. This means `loc_b` is adjacent to `rloc` in
           the opposite direction (`opp_dir`). Find `rloc` such that
           `adjacent(rloc, loc_b, dir)` holds. This is equivalent to finding `rloc`
           such that `(rloc, dir)` is in the reverse adjacency list of `loc_b`.
           If no such robot position exists, the path is impossible, return infinity.
        f. Calculate the shortest path distance for the robot from its current
           location (`robot_loc`) to the required push location `rloc`
           (`robot_approach_dist`). If the push position is unreachable by the robot,
           return infinity.
        g. The estimated cost for this box is `box_dist` (pushes) + `robot_approach_dist`
           (initial robot moves) + `max(0, box_dist - 1)` (estimated robot moves
           between subsequent pushes). Add this to the total heuristic value.
    4. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and building graph."""
        self.goals = task.goals
        static_facts = task.static

        # Build adjacency graphs and get list of all locations mentioned in adjacent facts
        self.adj_graph, self.adj_graph_reverse, self.locations = build_adjacency_graphs(static_facts)

        # Compute all-pairs shortest paths
        self.distances = compute_distances(self.locations, self.adj_graph)

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at":
                box, location = args
                self.goal_locations[box] = location

        # Store a large value for unreachable distances
        self.unreachable = float('inf')


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # Find current robot location
        robot_loc = None
        for fact in state:
            if match(fact, "at-robot", "*"):
                robot_loc = get_parts(fact)[1]
                break
        if robot_loc is None or robot_loc not in self.locations:
             # Robot location not found or not in the graph of known locations
             return self.unreachable # pragma: no cover


        # Find current box locations for boxes we care about (those in goals)
        box_locations = {}
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)
                if obj in self.goal_locations:
                     box_locations[obj] = loc

        total_cost = 0

        # For each box that needs to reach a goal
        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box)

            if current_box_loc is None or current_box_loc not in self.locations:
                 # Box location not found or not in the graph of known locations
                 return self.unreachable # pragma: no cover

            if current_box_loc == goal_loc:
                continue # Box is already at its goal

            # Calculate box distance to goal
            box_dist = self.distances.get((current_box_loc, goal_loc), self.unreachable)
            if box_dist == self.unreachable:
                # Goal is unreachable for the box
                return self.unreachable

            # Find a location loc_next adjacent to current_box_loc that is one step closer to goal_loc
            loc_next = None
            push_dir = None
            # Iterate through neighbors of current_box_loc
            for neighbor, direction in self.adj_graph.get(current_box_loc, []):
                 # Check if neighbor is closer to the goal
                 neighbor_dist_to_goal = self.distances.get((neighbor, goal_loc), self.unreachable)
                 if neighbor_dist_to_goal != self.unreachable and neighbor_dist_to_goal == box_dist - 1:
                     loc_next = neighbor
                     push_dir = direction
                     break # Found a suitable next step, pick the first one

            if loc_next is None:
                 # Cannot find a next step closer to the goal.
                 # This implies the box is stuck or the graph is weird.
                 # Given box_dist > 0 and finite, there should be a neighbor at dist-1.
                 # This case might indicate a deadlock not captured by simple distance,
                 # or an issue with the graph/distances. Treat as unreachable.
                 return self.unreachable # pragma: no cover


            # Find the required robot location (rloc) to push from current_box_loc to loc_next (direction push_dir)
            # Robot must be at rloc such that adjacent(rloc, current_box_loc, push_dir)
            # This means (rloc, push_dir) is in adj_graph_reverse[current_box_loc]
            rloc = None
            for source_loc, direction in self.adj_graph_reverse.get(current_box_loc, []):
                 if direction == push_dir:
                     rloc = source_loc
                     break # Found the required robot location

            if rloc is None or rloc not in self.locations:
                 # Cannot find the required robot push position in the graph.
                 # This path is impossible.
                 return self.unreachable # pragma: no cover


            # Calculate robot distance to the required push location
            robot_approach_dist = self.distances.get((robot_loc, rloc), self.unreachable)
            if robot_approach_dist == self.unreachable:
                # Robot cannot reach the required push position
                return self.unreachable # pragma: no cover

            # Add cost for this box:
            # box_dist (pushes)
            # + robot_approach_dist (initial robot moves to get behind the box)
            # + max(0, box_dist - 1) (estimated robot moves between subsequent pushes)
            total_cost += box_dist + robot_approach_dist + max(0, box_dist - 1)

        return total_cost
