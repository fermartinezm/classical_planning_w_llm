from heuristics.heuristic_base import Heuristic
from task import Task
from collections import deque
import re

# Helper function to parse PDDL fact strings
def parse_fact(fact_string):
    """
    Parses a PDDL fact string like '(predicate arg1 arg2)' into a tuple
    ('predicate', 'arg1', 'arg2').
    """
    # Remove parentheses and split by spaces
    # Handles facts like '(at-robot loc_6_4)', '(clear loc_2_4)', '(at box1 loc_4_4)', '(adjacent loc_4_2 loc_4_3 right)'
    # Assumes no spaces within predicate names or object names
    parts = fact_string[1:-1].split()
    return tuple(parts)

# Helper function to get opposite direction
def get_opposite_direction(direction):
    """Returns the opposite direction string."""
    if direction == 'down': return 'up'
    if direction == 'up': return 'down'
    if direction == 'left': return 'right'
    if direction == 'right': return 'left'
    return None # Should not happen for valid sokoban directions

# Helper function for BFS on the grid graph
def bfs(start_loc, graph, locations):
    """
    Performs BFS starting from start_loc on the given graph to find shortest
    path distances to all other locations. Returns a dictionary mapping
    locations to their distances from start_loc.
    """
    distances = {loc: float('inf') for loc in locations}
    distances[start_loc] = 0
    queue = deque([start_loc])

    while queue:
        current_loc = queue.popleft()
        dist = distances[current_loc]

        # Check if current_loc has neighbors in the graph dictionary
        if current_loc in graph:
            for neighbor in graph[current_loc]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = dist + 1
                    queue.append(neighbor)
    return distances


class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
    Estimates the cost to reach the goal state by summing the estimated
    number of pushes required for each box to reach its goal, plus the
    estimated number of robot moves required to reach a position from
    which it can make the first goal-reducing push for any box.
    The estimated number of pushes for a box is the shortest path distance
    on the grid graph (ignoring obstacles) from the box's current location
    to its goal location.
    The estimated number of robot moves is the shortest path distance on
    the grid graph (ignoring obstacles) from the robot's current location
    to a location adjacent to a box, on the opposite side of a potential
    goal-reducing push direction. The minimum such distance over all boxes
    and all goal-reducing push directions is used.

    Assumptions:
    - The problem defines a grid-like structure using 'adjacent' facts.
    - Locations are named using strings like 'loc_row_col'.
    - The goal specifies a fixed target location for each box using '(at boxX locY)'.
    - The grid graph derived from 'adjacent' facts is connected for relevant locations.
    - The heuristic is non-admissible and designed for greedy best-first search.

    Heuristic Initialization:
    1. Parses all 'adjacent' facts from the static information to build an
       undirected graph (for distance calculation) and a directed graph
       with directions (for finding push positions). Collects all unique
       location names (as strings like 'loc_row_col').
    2. Parses goal facts to create a mapping from box names to their target
       goal locations.
    3. Computes all-pairs shortest paths on the undirected grid graph using BFS.
       This precomputes the minimum number of robot moves (or box pushes on
       an empty grid) between any two locations, ignoring dynamic obstacles.

    Step-By-Step Thinking for Computing Heuristic:
    1. Identify the current location of the robot and each box from the state.
    2. Check if the goal is already reached. If yes, return 0.
    3. Initialize the total heuristic value `h` to 0.
    4. Initialize the minimum robot distance to any potential push position
       `min_robot_dist_to_any_push_pos` to infinity.
    5. Set a flag `any_box_needs_push` to False.
    6. Iterate through each box currently in the state:
        a. Get the box's current location and its target goal location (from precomputation).
        b. If the box is already at its goal, skip it.
        c. Set `any_box_needs_push` to True.
        d. Calculate the grid distance (estimated pushes) from the box's current
           location to its goal location using the precomputed distances.
        e. If the grid distance is infinity, the box cannot reach the goal even
           on an empty grid, indicating a dead end. Return infinity immediately.
        f. Add the box's grid distance to `h`.
        g. Find potential goal-reducing push directions for this box: Iterate
           through neighbors of the box's current location in the undirected graph.
           A neighbor is goal-reducing if moving the box there reduces its grid
           distance to the goal.
        h. For each goal-reducing neighbor location (`next_b_loc`):
            i. Determine the required robot position (`push_loc`) adjacent to the
               box's current location (`b_loc`), on the opposite side of the push
               direction (from `b_loc` to `next_b_loc`). This is found by looking
               up the direction in the directed graph and finding the neighbor
               of `b_loc` in the opposite direction.
            ii. If a valid `push_loc` exists in the grid:
                - Calculate the grid distance from the robot's current location
                  to this `push_loc` using the precomputed distances.
                - Update `min_robot_dist_to_any_push_pos` with the minimum
                  distance found so far across all goal-reducing pushes for
                  all boxes.
    7. After iterating through all boxes, if `any_box_needs_push` is True:
        a. If `min_robot_dist_to_any_push_pos` is still infinity, it means the
           robot cannot reach any position to make a goal-reducing push for
           any box (on the empty grid). This indicates a dead end. Return infinity.
        b. Add `min_robot_dist_to_any_push_pos` to `h`.
    8. Return the calculated value `h`. If no boxes needed pushing, `h` is 0,
       correctly indicating a goal state.
    """

    def __init__(self, task):
        super().__init__()
        self.task = task # Store task for access to goals etc.

        self.locations = set()
        self.graph = {}  # Undirected adjacency list: loc -> [neighbor_loc, ...]
        self.adj_dir_graph = {} # Directed adjacency list: loc -> [(neighbor_loc, direction), ...]
        self.box_goals_map = {} # box_name -> goal_loc

        # Parse static facts to build graphs and collect locations
        for fact_string in self.task.static:
            parts = parse_fact(fact_string)
            if parts[0] == 'adjacent':
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                self.locations.add(loc1)
                self.locations.add(loc2)

                # Undirected graph
                if loc1 not in self.graph: self.graph[loc1] = []
                if loc2 not in self.graph: self.graph[loc2] = []
                self.graph[loc1].append(loc2)
                self.graph[loc2].append(loc1)

                # Directed graph with directions
                if loc1 not in self.adj_dir_graph: self.adj_dir_graph[loc1] = []
                self.adj_dir_graph[loc1].append((loc2, direction))

        # Ensure all locations mentioned in initial state and goals are included
        # This handles cases where a location might be isolated (no adjacent facts)
        # but contains the robot/box/goal.
        all_relevant_facts = set(self.task.initial_state) | set(self.task.goals)
        for fact_string in all_relevant_facts:
             parts = parse_fact(fact_string)
             for part in parts:
                 if part.startswith('loc_'):
                     self.locations.add(part)

        # Parse goal facts to map boxes to goals
        for fact_string in self.task.goals:
            parts = parse_fact(fact_string)
            if parts[0] == 'at':
                box_name, goal_loc = parts[1], parts[2]
                self.box_goals_map[box_name] = goal_loc
                # Also add goal locations to locations set if not already there
                self.locations.add(goal_loc)

        # Compute all-pairs shortest paths on the undirected grid graph
        self.grid_dist = {}
        # Convert set to list for consistent iteration order if needed, though BFS handles sets fine
        location_list = list(self.locations)
        for loc in location_list:
            self.grid_dist[loc] = bfs(loc, self.graph, location_list)

    def __call__(self, node):
        state = node.state

        # Check if goal is reached (heuristic is 0)
        if self.task.goal_reached(state):
             return 0

        # Get current robot and box locations
        robot_loc = None
        box_locs_map = {}
        for fact_string in state:
            parts = parse_fact(fact_string)
            if parts[0] == 'at-robot':
                robot_loc = parts[1]
            elif parts[0] == 'at':
                box_name, current_loc = parts[1], parts[2]
                box_locs_map[box_name] = current_loc

        # Should always have a robot in a valid state in Sokoban
        if robot_loc is None:
             return float('inf') # Indicates an unexpected state

        h = 0
        min_robot_dist_to_any_push_pos = float('inf')
        any_box_needs_push = False

        # Iterate through boxes currently in the state
        for box_name, b_loc in box_locs_map.items():
            # Ensure this box is one we care about (has a goal)
            if box_name not in self.box_goals_map:
                 # This box is not in the goal state according to the task definition.
                 # We assume only boxes with goals are relevant for the heuristic.
                 continue

            b_goal = self.box_goals_map[box_name]

            if b_loc == b_goal:
                continue # Box is already at goal

            any_box_needs_push = True

            # Box distance (estimated pushes)
            # Check if locations exist in the precomputed distances
            if b_loc not in self.grid_dist or b_goal not in self.grid_dist.get(b_loc, {}):
                 # This indicates a location in state/goal was not in the graph
                 # derived from adjacent facts. Likely a malformed problem.
                 return float('inf')

            box_dist = self.grid_dist[b_loc][b_goal]

            if box_dist == float('inf'):
                # Box cannot reach goal location even on empty grid -> Dead end
                return float('inf')

            h += box_dist

            # Robot distance to a push position for this box
            # Find potential goal-reducing push directions for this box
            # Iterate through neighbors of b_loc in the undirected graph
            if b_loc in self.graph: # Check if b_loc has neighbors in the graph
                for next_b_loc in self.graph[b_loc]:
                    # Check if this push reduces the box's distance to the goal
                    # Use .get() for safety, although locations should be in self.locations
                    current_box_dist = self.grid_dist.get(b_loc, {}).get(b_goal, float('inf'))
                    next_box_dist = self.grid_dist.get(next_b_loc, {}).get(b_goal, float('inf'))

                    if next_box_dist < current_box_dist: # Goal-reducing push
                        # Find direction from b_loc to next_b_loc
                        direction_to_next = None
                        for neighbor, d in self.adj_dir_graph.get(b_loc, []):
                            if neighbor == next_b_loc:
                                direction_to_next = d
                                break

                        if direction_to_next:
                            # Find push_loc: neighbor of b_loc in opposite direction
                            opposite_dir = get_opposite_direction(direction_to_next)
                            push_loc = None
                            # Iterate through neighbors of b_loc in the directed graph to find the one in the opposite direction
                            for neighbor_of_b, d in self.adj_dir_graph.get(b_loc, []):
                                 if d == opposite_dir:
                                     push_loc = neighbor_of_b
                                     break

                            if push_loc:
                                # Calculate robot distance to this push_loc
                                if robot_loc in self.grid_dist and push_loc in self.grid_dist.get(robot_loc, {}):
                                    robot_dist = self.grid_dist[robot_loc][push_loc]
                                    # Update the minimum robot distance needed for any push
                                    min_robot_dist_to_any_push_pos = min(min_robot_dist_to_any_push_pos, robot_dist)
                                else:
                                    # Robot or push_loc not in graph? Malformed problem.
                                    return float('inf')

        # If there are boxes not at their goals, add the minimum robot distance needed
        if any_box_needs_push:
            if min_robot_dist_to_any_push_pos == float('inf'):
                 # No reachable push position found for any box that needs pushing
                 # This is a strong indicator of a dead end under the relaxed assumptions
                 return float('inf')
            h += min_robot_dist_to_any_push_pos

        # If no boxes needed pushing, h is 0, which is correct for goal state.
        return h
