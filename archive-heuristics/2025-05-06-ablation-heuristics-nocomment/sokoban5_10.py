from heuristics.heuristic_base import Heuristic
from collections import deque

# Helper function to parse PDDL facts
def get_parts(fact):
    """Removes surrounding parentheses and splits by space."""
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Sokoban domain.

    Summary:
        This heuristic estimates the cost to reach the goal state by summing
        the shortest path distances for each box to its goal location and
        the shortest path distance for the robot to reach each box that is
        not yet at its goal. It uses precomputed distances on the grid defined
        by the 'adjacent' facts. This heuristic is non-admissible but aims
        to guide the search towards states where boxes are closer to their
        goals and the robot is closer to the boxes that need moving.

    Assumptions:
        - The grid structure is defined by 'adjacent' facts, and adjacency is
          symmetric (if A is adjacent to B, B is adjacent to A).
        - Locations are consistently named.
        - Each box has a unique goal location specified in the task goals.
        - All locations mentioned in initial state, goals, or adjacent facts
          are part of a connected graph, or unreachable goals/boxes imply
          infinite cost.

    Heuristic Initialization:
        1. Parses the task goals to identify the target location for each box.
        2. Builds an undirected graph representing the grid connectivity based
           on the 'adjacent' facts from the static information. Includes all
           locations mentioned in initial state, goals, or adjacent facts.
        3. Precomputes the shortest path distance between all pairs of locations
           in the graph using Breadth-First Search (BFS). These distances
           represent the minimum number of 'move' actions between locations.

    Step-By-Step Thinking for Computing Heuristic:
        1. For a given state, identify the current location of the robot and
           the current location of each box that is a goal object.
        2. Initialize the total heuristic cost to 0.
        3. Iterate through each box and its corresponding goal location (as
           identified during initialization):
            a. Get the box's current location from the state.
            b. If the box is already at its goal location, it contributes 0 to
               the heuristic for this box. Continue to the next box.
            c. If the box is not at its goal location:
                i. Calculate the shortest path distance from the box's current
                   location to its goal location using the precomputed distances.
                   This estimates the minimum number of pushes required for this box
                   along a clear path. If the goal is unreachable from the box's
                   current location in the precomputed graph, the state is
                   considered to have infinite cost (likely unsolvable).
                ii. Calculate the shortest path distance from the robot's current
                    location to the box's current location using the precomputed
                    distances. This estimates the minimum number of robot moves
                    to reach the box. If the box is unreachable from the robot's
                    current location, return infinity.
                iii. Add the box-to-goal distance and the robot-to-box distance
                     to the total heuristic cost.
        4. Return the total accumulated cost.
    """
    def __init__(self, task):
        self.goals = task.goals
        static_facts = task.static

        # 1. Extract goal locations for each box
        self.goal_locations = {}
        for goal in self.goals:
            parts = get_parts(goal)
            # Goal is (at box location)
            if parts[0] == 'at' and len(parts) == 3:
                obj, location = parts[1], parts[2]
                # Assuming objects in goal 'at' predicates are boxes
                self.goal_locations[obj] = location

        # 2. Build the graph from adjacent facts and collect all relevant locations
        self.graph = {}
        all_locations_set = set()

        # Add locations from adjacent facts
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                all_locations_set.add(loc1)
                all_locations_set.add(loc2)
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Assuming symmetric adjacency

        # Add locations from initial state and goals that might not be in adjacent facts
        # (e.g., isolated locations, though unlikely in typical Sokoban grids)
        for fact in task.initial_state:
             parts = get_parts(fact)
             # Check for predicates that specify location
             if parts[0] in ['at-robot', 'at'] and len(parts) >= 2:
                 loc = parts[-1] # Location is the last argument
                 all_locations_set.add(loc)
                 if loc not in self.graph:
                     self.graph[loc] = set() # Add isolated node

        for loc in self.goal_locations.values():
             all_locations_set.add(loc)
             if loc not in self.graph:
                 self.graph[loc] = set() # Add isolated node

        self.all_locations = list(all_locations_set)

        # 3. Precompute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.all_locations:
            self.distances[start_loc] = {}
            queue = deque([(start_loc, 0)])
            visited = {start_loc}
            while queue:
                current_loc, dist = queue.popleft()
                self.distances[start_loc][current_loc] = dist
                # Use graph.get(current_loc, set()) in case a location was added
                # from initial/goal state but has no adjacent facts.
                for neighbor in self.graph.get(current_loc, set()):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

    def __call__(self, node):
        state = node.state

        # 1. Identify robot and box locations
        robot_location = None
        box_locations = {} # {box_name: location}
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at-robot' and len(parts) == 2:
                robot_location = parts[1]
            elif parts[0] == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                # Only track locations for boxes that are in the goals
                if obj in self.goal_locations:
                     box_locations[obj] = loc

        # Check if robot location was found (should always be the case in a valid state)
        if robot_location is None:
             # This indicates an invalid state representation or domain issue
             # For robustness, return infinity or a large value
             return float('inf')

        # 2. Initialize total cost
        total_cost = 0

        # 3. Iterate through boxes and calculate cost
        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locations.get(box)

            # This case should ideally not happen if box_locations is populated correctly
            # for all boxes in goal_locations. Add a check just in case.
            if current_loc is None:
                 # Box not found in state? Problematic state.
                 return float('inf')

            if current_loc != goal_loc:
                # i. Box distance to goal
                # Check if current_loc is a valid start location in our distance map
                if current_loc not in self.distances:
                    # This location was not in adjacent facts, initial state, or goals.
                    # Should not happen if all_locations_set is built correctly.
                    return float('inf')

                # Check if goal_loc is reachable from current_loc
                if goal_loc not in self.distances[current_loc]:
                    # Box cannot reach its goal location
                    return float('inf')
                box_dist = self.distances[current_loc][goal_loc]

                # ii. Robot distance to box
                # Check if robot_location is a valid start location
                if robot_location not in self.distances:
                     return float('inf') # Robot location not in graph?

                # Check if box_location is reachable from robot_location
                if current_loc not in self.distances.get(robot_location, {}):
                    # Robot cannot reach the box
                    return float('inf')
                robot_dist_to_box = self.distances[robot_location][current_loc]

                # iii. Add costs
                # Sum of box distance to goal and robot distance to box
                total_cost += box_dist + robot_dist_to_box

        # 4. Return total cost
        # If total_cost is 0, it means all boxes are at their goals.
        return total_cost
