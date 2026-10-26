import sys
from collections import deque
# Assuming the heuristic base class is available in the 'heuristics' package
# If the directory structure is different, adjust the import path as needed.
# e.g., from ..heuristics.heuristic_base import Heuristic
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts
def get_parts(fact: str):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(at box1 loc_1_1)" -> ["at", "box1", "loc_1_1"]
    Removes parentheses and splits by space. Handles simple cases.
    """
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal state in a Sokoban problem.
    It calculates the sum of the shortest path distances for each box from its current
    location to its target goal location. Additionally, it adds the shortest distance
    from the robot's current location to the nearest box that is not yet in its
    goal location. This encourages states where the robot is close to a box
    that needs to be moved, aiming for a more informed (though non-admissible) heuristic
    suitable for greedy best-first search.

    # Assumptions
    - Each box mentioned in the goal conditions (e.g., `(at box1 loc_goal)`) has a
      unique target goal location specified.
    - The cost of moving the robot ('move' action) or pushing a box ('push' action) is 1.
    - The heuristic ignores interactions between boxes (i.e., one box blocking another's path).
      It calculates distances assuming paths are clear except for permanent walls.
    - The heuristic ignores complex dead-end states (locations or configurations from
      which a box cannot reach its goal, e.g., corners unless they are goals). It only
      considers basic reachability through the grid structure defined by 'adjacent' facts.
    - The `adjacent` predicates in the static facts define the connectivity of the grid.
      Locations not mentioned or connected are considered inaccessible (walls).
    - Adjacency is symmetric: if `(adjacent l1 l2 dir)` exists, the underlying grid allows
      movement between l1 and l2. The BFS explores this bidirectionally.

    # Heuristic Initialization
    - Parses the task's static facts (`adjacent`) to build an undirected graph
      representation of the accessible locations (grid map). Stores the set of all unique
      locations and an adjacency list (`self.adj`).
    - Identifies the target goal location for each box from the task goals (`at` predicates
      involving objects assumed to be boxes). Stores this in `self.goal_locations`.
    - Stores the set of boxes relevant to the goal (`self.boxes`).
    - Precomputes all-pairs shortest path distances between all accessible locations using
      Breadth-First Search (BFS). The distance `dist(l1, l2)` represents the minimum
      number of steps (robot moves or box pushes along clear paths) required to get
      from location `l1` to `l2`. Stores infinity (`float('inf')`) for unreachable pairs
      in `self.distances`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Parse State:** Extract the current robot location (`robot_loc`) and the
        location of each goal-relevant box (`box_locs`) from the input state `node.state`.
        Perform sanity checks: ensure robot location is found and is a known, valid location;
        ensure all goal boxes have locations specified in the state. Return infinity if checks fail,
        as the state might be invalid or inconsistent with the precomputed map.
    2.  **Goal Check:** Verify if the current state satisfies all goal conditions. Check if
        every box `b` in `self.goal_locations` is at its corresponding target location
        `self.goal_locations[b]` according to `box_locs`. If yes, the heuristic value is 0.
    3.  **Sum Box Distances:** Initialize `sum_box_distances = 0`. Iterate through each
        box `b` tracked in `self.goal_locations`. Find its current location `current_loc`
        from `box_locs`. If `current_loc` is not the `goal_loc`:
        - Perform sanity checks: ensure `current_loc` is a known location and distance data exists.
        - Retrieve the precomputed shortest path distance `dist = self.distances[current_loc].get(goal_loc, self.infinity)`.
        - If `dist` is infinity, the box cannot reach its goal from its current position based
          on grid connectivity alone. The state is considered unsolvable by this heuristic. Return infinity.
        - Otherwise, add `dist` to `sum_box_distances`. Keep track of boxes that are misplaced
          and their locations in `misplaced_boxes_locs`.
    4.  **Min Robot Distance:** Initialize `min_robot_dist = self.infinity`. If there are no
        misplaced boxes (goal state achieved), return 0 (this case is handled by step 2).
        Otherwise, iterate through the `misplaced_boxes_locs`. For each misplaced box `b` at
        `box_loc`, retrieve the precomputed distance from the robot's current location
        `robot_loc` to the box's location: `dist = self.distances[robot_loc].get(box_loc, self.infinity)`.
        Update `min_robot_dist = min(min_robot_dist, dist)`.
        - Perform sanity check: ensure distance data exists for `robot_loc`.
        - If after checking all misplaced boxes, `min_robot_dist` remains infinity, it means
          the robot cannot reach any of the boxes that need moving (e.g., robot is trapped).
          Return infinity.
    5.  **Combine Costs:** The final heuristic estimate is `h = sum_box_distances + min_robot_dist`.
        This value represents the estimated total pushes needed (sum of box distances) plus
        an estimate of the effort for the robot to get close to the nearest box that needs attention.
    6.  **Return Value:** Return the calculated heuristic value `h`. It will be 0 for goal
        states, a positive integer for reachable non-goal states, and infinity for states
        detected as potentially unsolvable based on grid reachability.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by processing static information (adjacencies, goals)
        and precomputing all-pairs shortest path distances between locations.
        """
        self.goals = task.goals
        static_facts = task.static
        self.locations = set()
        self.adj = {}  # Adjacency list: {loc: List[neighbor]}
        self.goal_locations = {}  # Target location for each box: {box_name: goal_loc}
        self.boxes = set()  # Set of box names relevant to the goal
        self.infinity = float('inf')

        # 1. Parse static 'adjacent' facts to build the graph structure
        adj_sets = {} # Use sets temporarily to handle potential duplicate facts
        for fact in static_facts:
            parts = get_parts(fact)
            # Format: (adjacent ?l1 - location ?l2 - location ?d - direction)
            if parts[0] == 'adjacent' and len(parts) == 4:
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                # Add edges assuming the underlying grid connectivity is symmetric
                adj_sets.setdefault(l1, set()).add(l2)
                adj_sets.setdefault(l2, set()).add(l1)

        # Convert adjacency sets to lists for consistent iteration (optional but good practice)
        for loc, neighbors in adj_sets.items():
            self.adj[loc] = list(neighbors)

        # 2. Parse goals to find target locations for boxes
        for goal in self.goals:
            parts = get_parts(goal)
            # Format: (at ?o - box ?l - location)
            if parts[0] == 'at' and len(parts) == 3:
                box, loc = parts[1], parts[2]
                # Assume objects in 'at' goals are the boxes we need to track.
                # A more robust approach might use PDDL typing info if available.
                self.goal_locations[box] = loc
                self.boxes.add(box)
                # Ensure goal locations are part of our known locations
                self.locations.add(loc)

        # Ensure all locations mentioned in adjacency lists are included in self.locations
        # This handles cases where a location might only appear as a neighbor
        all_adj_locations = set(self.adj.keys())
        for neighbors in self.adj.values():
            all_adj_locations.update(neighbors)
        self.locations.update(all_adj_locations)

        # 3. Precompute all-pairs shortest paths using BFS
        self.distances = self._compute_all_pairs_shortest_paths()


    def _bfs(self, start_node):
        """
        Performs Breadth-First Search starting from start_node to find the
        shortest path distance to all reachable locations within the known grid.
        Returns a dictionary mapping {location: distance}.
        """
        # Initialize distances to infinity for all known locations
        distances = {loc: self.infinity for loc in self.locations}

        # If the start node is not a valid location in our map, return infinite distances
        if start_node not in self.locations:
            return distances

        distances[start_node] = 0
        queue = deque([start_node])

        while queue:
            current_loc = queue.popleft()
            current_dist = distances[current_loc]

            # Explore neighbors using the pre-built adjacency list
            for neighbor in self.adj.get(current_loc, []):
                # If neighbor is a known location and we haven't found a path yet
                # (BFS guarantees the first path found is the shortest)
                if neighbor in distances and distances[neighbor] == self.infinity:
                    distances[neighbor] = current_dist + 1
                    queue.append(neighbor)
        return distances

    def _compute_all_pairs_shortest_paths(self):
        """
        Computes shortest path distances between all pairs of known locations by running
        BFS starting from each location.
        Returns a dictionary of dictionaries: {loc1: {loc2: distance}}.
        """
        all_distances = {}
        for loc in self.locations:
            # Run BFS from each location to find distances to all others
            all_distances[loc] = self._bfs(loc)
        return all_distances

    def __call__(self, node):
        """
        Calculates the heuristic value for the given state node.
        Returns an estimate of the cost to reach the goal, or infinity if the
        state seems unsolvable based on reachability.
        """
        state = node.state
        robot_loc = None
        box_locs = {}  # Current location of each goal-relevant box: {box_name: current_loc}

        # 1. Parse current state for robot and relevant box locations
        for fact in state:
            parts = get_parts(fact)
            predicate = parts[0]
            if predicate == 'at-robot' and len(parts) == 2:
                robot_loc = parts[1]
            elif predicate == 'at' and len(parts) == 3:
                obj, loc = parts[1], parts[2]
                if obj in self.boxes: # Only track boxes relevant to our goals
                    box_locs[obj] = loc

        # --- Sanity Checks ---
        if not robot_loc: return self.infinity # Robot location missing in state
        if robot_loc not in self.locations: return self.infinity # Robot is at an unknown/invalid location
        if len(box_locs) != len(self.boxes): return self.infinity # State doesn't contain locations for all goal boxes

        # 2. Check if goal is reached
        is_goal = True
        for box, goal_loc in self.goal_locations.items():
            # box_locs check above ensures box is present
            if box_locs[box] != goal_loc:
                is_goal = False
                break
        if is_goal:
            return 0 # Goal state has 0 cost

        # 3. Calculate sum of box distances to their goals
        sum_box_distances = 0
        misplaced_boxes_locs = {}  # Store locations of misplaced boxes {box: loc}

        for box, goal_loc in self.goal_locations.items():
            current_loc = box_locs[box]
            # Sanity check: Box location must be known
            if current_loc not in self.locations: return self.infinity

            if current_loc != goal_loc:
                # Check if distance data exists for the current location
                if current_loc not in self.distances: return self.infinity
                # Retrieve precomputed distance
                dist = self.distances[current_loc].get(goal_loc, self.infinity)

                # If distance is infinite, the goal is unreachable for this box
                if dist == self.infinity: return self.infinity
                sum_box_distances += dist
                misplaced_boxes_locs[box] = current_loc

        # 4. Calculate min robot distance to a misplaced box location
        min_robot_dist = self.infinity
        # If there are no misplaced boxes, goal should have been detected.
        # If somehow missed, this check prevents errors.
        if not misplaced_boxes_locs: return 0

        # Check if distance data exists for the robot's location
        if robot_loc not in self.distances: return self.infinity

        for box, box_loc in misplaced_boxes_locs.items():
             # box_loc is guaranteed to be in self.locations from check in step 3
             dist = self.distances[robot_loc].get(box_loc, self.infinity)
             min_robot_dist = min(min_robot_dist, dist)

        # If robot cannot reach any misplaced box, the state is likely unsolvable
        if min_robot_dist == self.infinity: return self.infinity

        # 5. Combine Costs: Sum of box-goal distances + min robot-box distance
        heuristic_value = sum_box_distances + min_robot_dist

        # 6. Return Value (guaranteed non-negative if not infinity)
        return heuristic_value
