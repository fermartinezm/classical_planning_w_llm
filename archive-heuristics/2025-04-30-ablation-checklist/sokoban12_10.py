from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at box1 loc_1_1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state by summing:
    1. The minimum number of pushes required for each box to reach its goal location
       (estimated by the shortest path distance on the static grid).
    2. The minimum number of robot moves required to reach a position from which
       it can make a progressive push on any box that is not yet at its goal.

    # Assumptions
    - The grid structure is defined by `adjacent` facts.
    - Shortest paths are computed on the static grid, ignoring dynamic objects (boxes, robot).
    - Goals are specified as `(at box location)`.

    # Heuristic Initialization
    - Parses goal conditions to map each box to its goal location.
    - Builds a graph representation of the grid from `adjacent` facts.
    - Precomputes all-pairs shortest paths on this static grid using BFS.
    - Stores a mapping of directions to their opposites.

    # Step-by-Step Thinking for Computing Heuristic
    1. Extract the current location of the robot and all boxes from the state.
    2. Initialize the heuristic value `h` to 0.
    3. Initialize `min_robot_dist_to_push_pos` to infinity. This will track the minimum distance
       from the robot's current location to *any* location from which it could make a
       progressive push on *any* box not at its goal.
    4. Iterate through each box and its goal location:
       - If the box is not at its goal location:
         - Add the precomputed shortest path distance from the box's current location
           to its goal location to `h`. This estimates the minimum number of pushes needed
           for this box, ignoring robot movement and obstacles.
         - Identify potential "progressive" push directions for this box: these are
           directions from the box's current location towards a neighbor location that
           is strictly closer to the box's goal location (based on precomputed distances).
         - For each progressive push direction:
           - Determine the required location for the robot: this is the location adjacent
             to the box's current location in the *opposite* direction of the push.
           - Calculate the precomputed shortest path distance from the robot's current
             location to this required robot location.
           - Update `min_robot_dist_to_push_pos` with the minimum distance found so far
             across all progressive pushes for all boxes not at their goals.
    5. If there are any boxes not at their goals and a reachable progressive push position
       was found for at least one of them, add `min_robot_dist_to_push_pos` to `h`.
       If boxes need moving but no reachable progressive push position exists, the state
       is likely unsolvable, return infinity.
    6. Return the calculated `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goals, building the grid graph,
        and precomputing shortest paths.
        """
        self.goals = task.goals
        static_facts = task.static

        # Store goal locations for each box.
        self.goal_locations = {}
        for goal in self.goals:
            predicate, *args = get_parts(goal)
            if predicate == "at" and len(args) == 2: # Ensure it's (at box location)
                box, location = args
                self.goal_locations[box] = location

        # Build graph and reverse graph from adjacent facts
        # Graph: loc -> list of (neighbor_loc, direction)
        # Reverse Graph: loc -> list of (location_behind_it, direction_from_behind_loc_to_loc)
        self.graph = {}
        self.reverse_graph = {}
        self.locations = set()

        for fact in static_facts:
            if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)

                if loc1 not in self.graph:
                    self.graph[loc1] = []
                self.graph[loc1].append((loc2, direction))

                if loc2 not in self.reverse_graph:
                     self.reverse_graph[loc2] = []
                # loc1 is adjacent to loc2 in 'direction'.
                # To push from loc1 to loc2, robot needs to be at a location 'behind' loc1.
                # The reverse graph helps find the location adjacent to loc1 from which a push *into* loc1 would come.
                # This is not quite what we need for robot_req_pos.
                # Let's rebuild reverse_graph to map loc -> list of (neighbor_loc, direction_from_neighbor_to_loc)
                # This is the same as the graph, just storing the opposite direction? No.
                # Let's store loc -> list of (neighbor_loc, direction_from_loc_to_neighbor) in graph
                # And loc -> list of (neighbor_loc, direction_from_neighbor_to_loc) in reverse_graph
                # The adjacent fact (loc1 loc2 dir) means loc1 -> loc2 is dir, and loc2 -> loc1 is opposite(dir).
                # Let's just store the graph and use opposite_direction map.

        # Rebuild graph storing loc -> {neighbor_loc: direction} for easier lookup
        self.graph = {}
        for fact in static_facts:
             if match(fact, "adjacent", "*", "*", "*"):
                _, loc1, loc2, direction = get_parts(fact)
                self.locations.add(loc1)
                self.locations.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = {}
                self.graph[loc1][loc2] = direction

        # Compute all-pairs shortest paths on the static grid
        self.distances = {}
        for loc in self.locations:
             self.distances[loc] = self._bfs(self.graph, loc)

        # Map directions to their opposites
        self.opposite_direction = {"up": "down", "down": "up", "left": "right", "right": "left"}


    def _bfs(self, graph, start_node):
        """
        Performs Breadth-First Search from a start node to find distances
        to all reachable nodes in the graph.
        """
        distances = {node: float('inf') for node in self.locations}
        if start_node not in distances:
             # This should not happen if self.locations is built correctly
             # from the graph nodes, but as a safeguard:
             return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_node = queue.popleft()

            if current_node in graph:
                # graph stores {neighbor: direction}
                for neighbor in graph[current_node]:
                    if distances[neighbor] == float('inf'):
                        distances[neighbor] = distances[current_node] + 1
                        queue.append(neighbor)
        return distances


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        to reach the goal state.
        """
        state = node.state

        robot_loc = None
        box_locations = {}
        # clear_locations = set() # Not directly needed for this heuristic calculation

        # Extract current state information
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at-robot":
                robot_loc = parts[1]
            elif parts[0] == "at" and len(parts) == 3:
                box, loc = parts[1], parts[2]
                box_locations[box] = loc
            # elif parts[0] == "clear":
            #      clear_locations.add(parts[1]) # Not used in this heuristic

        h = 0
        min_robot_dist_to_push_pos = float('inf')
        any_box_needs_move = False

        # Calculate sum of box-to-goal distances and find minimum robot distance to a push position
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)

            if current_loc is None:
                 # Should not happen in a valid Sokoban state - every box must be somewhere
                 # Treat as unsolvable from here? Or maybe the box is not part of this problem instance?
                 # Assuming valid instances where all goal boxes exist and are located.
                 continue # Skip if box location isn't found

            if current_loc != goal_loc:
                any_box_needs_move = True

                # Add box-to-goal distance (minimum pushes)
                if current_loc in self.distances and goal_loc in self.distances[current_loc]:
                    box_dist = self.distances[current_loc][goal_loc]
                    if box_dist == float('inf'):
                         # Box is in a location from which the goal is unreachable on the static grid
                         return float('inf') # This state is likely unsolvable
                    h += box_dist
                else:
                    # Should not happen if locations are consistent and in the graph
                    # print(f"Warning: Distance not found for box {box} from {current_loc} to {goal_loc}")
                    return float('inf') # Indicate problem

                # Find potential robot positions to make a progressive push on this box
                if current_loc in self.graph:
                    for neighbor_loc, push_dir in self.graph[current_loc].items():
                        # Check if pushing towards neighbor_loc reduces the box-goal distance
                        # Need to ensure neighbor_loc is reachable from goal_loc for distance comparison
                        if neighbor_loc in self.distances and goal_loc in self.distances[neighbor_loc]:
                             if self.distances[current_loc][goal_loc] > self.distances[neighbor_loc][goal_loc]:
                                # This is a progressive push direction (current_loc -> neighbor_loc)
                                # Robot needs to be at robot_req_pos, adjacent to current_loc in opposite direction
                                required_dir = self.opposite_direction.get(push_dir)
                                if required_dir:
                                    # Find the location adjacent to current_loc in the required_dir
                                    # We need to search through all locations to find one whose 'push_dir' neighbor is current_loc
                                    # and the direction is the required_dir.
                                    robot_req_pos = None
                                    if current_loc in self.graph: # Check locations that can push *into* current_loc
                                         for potential_robot_loc, dir_from_potential in self.graph.items():
                                             if current_loc in dir_from_potential and dir_from_potential[current_loc] == required_dir:
                                                 robot_req_pos = potential_robot_loc
                                                 break # Found the required position behind the box

                                    if robot_req_pos and robot_loc in self.distances and robot_req_pos in self.distances[robot_loc]:
                                        robot_dist = self.distances[robot_loc][robot_req_pos]
                                        min_robot_dist_to_push_pos = min(min_robot_dist_to_push_pos, robot_dist)
                                    # else: robot_loc or robot_req_pos not in graph? Problematic.
                                    # If robot_req_pos is None, it means there's no location from which
                                    # the robot can push into current_loc from the required direction.
                                    # This might indicate a wall or boundary issue, or a complex blockage.
                                    # If min_robot_dist_to_push_pos remains inf, it's handled later.


        # Add the minimum robot travel cost if any box needs moving and a reachable push position exists
        if any_box_needs_move:
            if min_robot_dist_to_push_pos != float('inf'):
                h += min_robot_dist_to_push_pos
            else:
                 # Boxes need moving, but robot cannot reach any position to make a progressive push
                 # This state is likely unsolvable
                 return float('inf')

        return h

