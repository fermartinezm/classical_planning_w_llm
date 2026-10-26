from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts gracefully
    if not fact or not isinstance(fact, str) or len(fact) < 2 or fact[0] != '(' or fact[-1] != ')':
        return []
    return fact[1:-1].split()

# Assume Heuristic base class is available from heuristics.heuristic_base
# class sokobanHeuristic(Heuristic):
class sokobanHeuristic:
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest path distances
    for each box from its current location to its assigned goal location.

    # Assumptions:
    - Each box has a unique goal location assigned in the problem definition.
    - The shortest path distance on the grid graph is a reasonable approximation of the number of pushes required for a box.
    - The cost of robot movement to get into position for pushes is implicitly handled or ignored for simplicity (non-admissible).
    - The heuristic does not detect dead-end states (e.g., boxes in corners).

    # Heuristic Initialization
    - Builds a graph representing the connectivity between locations based on `adjacent` facts.
    - Extracts the goal location for each box from the task's goal conditions.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Build the location graph: Iterate through static facts, identify `adjacent` predicates, and create an adjacency list representation where locations are nodes and adjacency represents edges. Since adjacency is bidirectional, add edges in both directions.
    2.  Map boxes to goal locations: Iterate through the task's goal conditions, identify `at` predicates for boxes, and store the target location for each box.
    3.  In the heuristic function (`__call__`):
        a.  Get the current state.
        b.  Find the current location of each box by checking `at` predicates in the state.
        c.  Initialize total heuristic cost to 0.
        d.  For each box identified in the goal mapping:
            i.  Get the box's current location and its goal location.
            ii. If the box is not at its goal location:
                - Compute the shortest path distance between the box's current location and its goal location using BFS on the location graph.
                - If no path exists (distance is infinity), return a very large value to indicate a likely unsolvable state.
                - Add this distance to the total heuristic cost.
        e.  Return the accumulated cost. If the cost is 0, it means all boxes are at their goal locations.
    """

    def __init__(self, task):
        """Initialize the heuristic by building the location graph and mapping goals."""
        static_facts = task.static

        # Build the location graph from adjacent facts
        self.graph = {}
        for fact in static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "adjacent" and len(parts) == 4:
                loc1, loc2, direction = parts[1:]
                if loc1 not in self.graph:
                    self.graph[loc1] = set()
                if loc2 not in self.graph:
                    self.graph[loc2] = set()
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Adjacency is bidirectional

        # Map boxes to their goal locations
        self.goal_locations = {}
        # Assuming task.goals is a frozenset of goal fact strings
        for goal in task.goals:
            parts = get_parts(goal)
            # Goal facts are typically (at ?box ?location)
            if parts and parts[0] == "at" and len(parts) == 3:
                 obj_name, loc_name = parts[1:]
                 # We assume any 'at' goal involving an object that is not the robot
                 # is a box goal. The domain definition confirms 'at' is for boxes.
                 self.goal_locations[obj_name] = loc_name


    def get_shortest_path_distance(self, start_loc, end_loc):
        """
        Computes the shortest path distance between two locations using BFS.
        Returns float('inf') if no path exists or locations are invalid.
        """
        if start_loc == end_loc:
            return 0
        # Ensure both locations exist in the graph
        if start_loc not in self.graph or end_loc not in self.graph:
             return float('inf')

        queue = deque([(start_loc, 0)])
        visited = {start_loc}

        while queue:
            current_loc, dist = queue.popleft()

            if current_loc == end_loc:
                return dist

            # Check if current_loc has neighbors in the graph
            if current_loc in self.graph:
                for neighbor in self.graph[current_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))

        return float('inf') # No path found


    def __call__(self, node):
        """Estimate the minimum cost to move all boxes to their goal locations."""
        state = node.state

        # Find current box locations and robot location
        current_box_locations = {}
        robot_location = None # Robot location is not used in this specific heuristic calculation, but finding it is part of state inspection.

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            if parts[0] == "at" and len(parts) == 3: # (at ?o ?l)
                 obj_name, loc_name = parts[1:]
                 # Only track locations for objects that are boxes according to our goals
                 if obj_name in self.goal_locations:
                      current_box_locations[obj_name] = loc_name
            elif parts[0] == "at-robot" and len(parts) == 2: # (at-robot ?l)
                 robot_location = parts[1]

        # Basic validation: Check if we found locations for all boxes expected from goals
        # and if the robot location was found. If not, this state is likely invalid or a dead end.
        if len(current_box_locations) != len(self.goal_locations) or robot_location is None:
             return float('inf')


        total_cost = 0

        # Calculate cost for each box that is not at its goal
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)

            # This check should ideally not be needed if the previous validation passes,
            # but provides robustness.
            if current_loc is None:
                 return float('inf')

            if current_loc != goal_loc:
                # Distance for the box to reach its goal
                box_dist = self.get_shortest_path_distance(current_loc, goal_loc)

                if box_dist == float('inf'):
                    # Box cannot reach its goal - likely a dead end
                    return float('inf')

                total_cost += box_dist

                # This heuristic only considers box-goal distance.
                # Robot movement cost is implicitly ignored or assumed minimal.

        return total_cost
