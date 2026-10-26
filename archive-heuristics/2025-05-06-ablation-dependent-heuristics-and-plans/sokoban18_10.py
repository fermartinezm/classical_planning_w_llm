# Assuming the Heuristic base class is available in heuristics.heuristic_base
from heuristics.heuristic_base import Heuristic
from fnmatch import fnmatch
from collections import defaultdict, deque

# Helper functions from Logistics example (re-used as they are general)
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

# New helper functions for Sokoban
def opposite_direction(direction_str):
    """Returns the opposite direction string."""
    if direction_str == 'up': return 'down'
    if direction_str == 'down': return 'up'
    if direction_str == 'left': return 'right'
    if direction_str == 'right': return 'left'
    return None # Should not happen in this domain

def bfs(graph, start, end, large_value=1000):
    """
    Performs BFS on the graph to find the shortest path distance.
    Returns distance or large_value if unreachable.
    """
    if start == end:
        return 0
    
    # Ensure start node exists in the graph before starting BFS
    # If start is not in the graph, it's unreachable from anywhere (except itself)
    if start not in graph:
         return large_value

    q = deque([(start, 0)])
    visited = {start}
    
    while q:
        curr_loc, dist = q.popleft()

        if curr_loc == end:
            return dist

        # Neighbors are locations reachable from curr_loc in the graph
        for neighbor in graph.get(curr_loc, []):
            if neighbor not in visited:
                visited.add(neighbor)
                q.append((neighbor, dist + 1))

    # If BFS completes without reaching the end
    return large_value # Large constant for unreachable

def build_sokoban_graphs(static_facts):
    """
    Builds the LocationGraph (for robot movement) and PushGraph (for box movement).
    """
    location_graph = defaultdict(set)
    push_graph = defaultdict(set)
    adjacent_map = {} # (l1, dir) -> l2
    reverse_adjacent_map = {} # (l2, dir) -> l1

    # First pass: Build basic adjacency maps
    for fact in static_facts:
        parts = get_parts(fact)
        if parts[0] == 'adjacent':
            _, l1, l2, dir = parts
            
            # Location graph (undirected for robot)
            location_graph[l1].add(l2)
            location_graph[l2].add(l1)

            # Adjacent maps (directed)
            adjacent_map[(l1, dir)] = l2
            # Handle cases where opposite direction might not be explicitly stated?
            # PDDL usually defines both directions if movement is bidirectional.
            # Let's assume opposite_direction(dir) always maps to a valid key if the link is bidirectional.
            opp_dir = opposite_direction(dir)
            if opp_dir: # Check if opposite_direction returned a valid direction
                 reverse_adjacent_map[(l2, opp_dir)] = l1


    # Second pass: Build PushGraph
    # A box can be pushed from l1 to l2 if (adjacent l1 l2 dir) AND
    # there exists l0 such that (adjacent l0 l1 dir) (robot position)
    for (l1, dir), l2 in adjacent_map.items():
        # Check if there's a location l0 such that robot can be at l0
        # and push from l1 to l2 in direction dir.
        # This l0 must be adjacent to l1 in the same direction dir.
        # So, l1 is adjacent to l0 in the opposite direction.
        required_robot_loc = reverse_adjacent_map.get((l1, opposite_direction(dir)))
        
        if required_robot_loc is not None:
             # If such a robot position exists, a push from l1 to l2 is possible
             push_graph[l1].add(l2)

    return location_graph, push_graph


class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing two components for each box
    that is not yet at its goal location:
    1. The minimum number of pushes required to move the box from its current location
       to its goal location, assuming the robot can instantly get into position for each push.
       This is calculated as the shortest path distance in a 'Push Graph'.
    2. The minimum number of robot moves required to get from the robot's current location
       to the box's current location. This is calculated as the shortest path distance
       in the grid's 'Location Graph'.

    # Assumptions
    - The grid structure and push possibilities are static and defined by 'adjacent' facts.
    - The heuristic uses static graph distances, ignoring dynamic 'clear' predicates
      for intermediate locations during path calculation. This makes it non-admissible
      but faster to compute. Unreachable locations in the static graph are assigned a large cost (1000).
    - The cost of a 'move' action and a 'push' action are both considered 1 for distance calculation purposes.

    # Heuristic Initialization
    - Extracts goal locations for each box from the task goals.
    - Builds two static graph representations of the grid from 'adjacent' facts:
        - A 'Location Graph' where edges represent adjacent locations (robot movement).
        - A 'Push Graph' where a directed edge u -> v exists if a box can be pushed
          from location u to location v (requires robot to be adjacent to u in the push direction).
    - Stores all unique location strings (implicitly within graph keys).

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the robot's current location.
    2. Identify the current location of each box.
    3. Initialize the total heuristic value to 0.
    4. For each box defined in the problem goals:
       a. Determine its goal location from the pre-calculated goal map.
       b. Determine its current location from the current state.
       c. If the box is already at its goal location, add 0 cost for this box and continue.
       d. If the box is not at its goal:
          i. Calculate the minimum pushes needed: Find the shortest path distance
             from the box's current location to its goal location in the 'Push Graph'.
             Use a large constant (1000) if the goal is unreachable in the Push Graph (e.g., box is in a corner).
          ii. Calculate the minimum robot moves needed to reach the box: Find the shortest
              path distance from the robot's current location to the box's current location
              in the 'Location Graph'. Use a large constant (1000) if unreachable.
          iii. Add the sum of the push distance and the robot distance to the total heuristic.
    5. Return the total heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building graphs.
        """
        # Assuming task.goals is a frozenset of goal facts like `(at box1 loc_R_C)`
        self.goals = task.goals
        self.static = task.static

        # Store goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 # Assuming goals only specify box locations and objects starting with 'box' are boxes
                 if obj.startswith('box'):
                      self.goal_locations[obj] = loc

        # Build static graphs from adjacent facts
        self.location_graph, self.push_graph = build_sokoban_graphs(self.static)

        # Collect all possible locations from the graphs for potential checks
        self.all_locations = set(self.location_graph.keys()).union(set(l for neighbors in self.location_graph.values() for l in neighbors))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state (frozenset of facts)

        # Find robot's current location
        robot_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]
                break
        
        # If robot location is not found or is not a known location, something is wrong.
        # Assign a high cost.
        if robot_location is None or robot_location not in self.all_locations:
             return 1000 # Large penalty

        # Find current locations of all boxes
        box_locations = {} # {box_name: loc_str}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and len(parts) == 3:
                 obj, loc = parts[1], parts[2]
                 if obj.startswith('box'): # Simple check for box type
                      box_locations[obj] = loc

        total_heuristic = 0
        large_penalty = 1000 # Define large penalty constant

        # For each box that has a goal location
        for box, goal_loc in self.goal_locations.items():
            current_box_loc = box_locations.get(box)

            # If a box specified in the goal is not found in the current state,
            # or its location is not a known location, it's likely an invalid/unsolvable state.
            if current_box_loc is None or current_box_loc not in self.all_locations:
                 return large_penalty # Large penalty

            # If box is already at its goal, cost is 0 for this box
            if current_box_loc == goal_loc:
                continue

            # Calculate box push distance (min pushes)
            # BFS on the push graph from current box location to goal location
            box_push_dist = bfs(self.push_graph, current_box_loc, goal_loc, large_value=large_penalty)

            # Calculate robot distance to the box (min robot moves)
            # BFS on the location graph from robot location to current box location
            robot_dist_to_box = bfs(self.location_graph, robot_location, current_box_loc, large_value=large_penalty)

            # If either distance is the large penalty, the state is likely problematic
            # (e.g., box goal unreachable in static push graph, robot cannot reach box in static location graph).
            if box_push_dist >= large_penalty or robot_dist_to_box >= large_penalty:
                 return large_penalty # Return large penalty immediately

            # Add costs for this box
            total_heuristic += box_push_dist + robot_dist_to_box

        # The heuristic is 0 if and only if all boxes are at their goal locations.
        # This happens when the loop finishes without adding any cost for misplaced boxes.
        # If total_heuristic is 0, it means all boxes are at goals.

        return total_heuristic
