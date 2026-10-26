from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class sokobanHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Sokoban domain.

    # Summary
    This heuristic estimates the cost to reach the goal by summing the shortest
    path distances for each box from its current location to its goal location.
    The shortest path distance is computed on the graph of locations defined
    by the 'adjacent' predicates. This heuristic is non-admissible.

    # Assumptions
    - The primary cost driver is moving boxes to their goal locations.
    - The heuristic ignores the robot's position and the cost of moving the robot
      to get into a pushing position.
    - The heuristic ignores potential deadlocks where boxes block each other
      or the robot, or are pushed into corners/walls from which they cannot
      be moved towards the goal.
    - The heuristic assumes the location graph defined by 'adjacent' predicates
      is static and can be precomputed.
    - The 'adjacent' predicates define a graph where movement is effectively
      bidirectional between two locations if there are 'adjacent' facts
      connecting them in both directions (e.g., A to B and B to A). The
      graph is treated as undirected for distance calculation.

    # Heuristic Initialization
    - Parses the goal conditions to map each box to its target location.
    - Builds an undirected graph of locations based on the 'adjacent' predicates
      from static facts.
    - Precomputes the shortest path distance between all pairs of locations
      using Breadth-First Search (BFS). Stores these distances in a dictionary
      `self.distances[start_loc][end_loc]`. Unreachable locations have a distance
      of `float('inf')`.
    - Calculates a large penalty value (`self.unreachable_penalty`) to be returned
      if any box's goal location is unreachable from its current location,
      indicating a likely unsolvable state or deadlock.

    # Step-By-Step Thinking for Computing Heuristic
    1. Get the current state of the world from the provided `node`.
    2. Identify the current location of each box by iterating through the state
       facts and finding predicates of the form `(at ?box ?location)`.
    3. Initialize the total heuristic cost (`total_heuristic`) to 0.
    4. For each box that has a specified goal location in the task:
       a. Retrieve the box's current location and its goal location.
       b. If the box's current location is not found in the state (should not
          happen in valid states) or if the box is not already at its goal location:
          i. Look up the precomputed shortest path distance from the box's
             current location to its goal location in the `self.distances` table.
          ii. If the distance is `float('inf')` (meaning the goal location is
              unreachable from the box's current location in the location graph),
              return the large constant `self.unreachable_penalty`. This signals
              that the state is likely a deadlock or unsolvable.
          iii. Otherwise (if the distance is finite), add this distance to the
               `total_heuristic`.
    5. Return the final `total_heuristic` value. If all boxes were already at
       their goal locations, the sum will be 0.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions and building
        the location graph for distance calculations.
        """
        super().__init__(task) # Call the base class constructor

        self.goals = task.goals
        static_facts = task.static

        # 1. Parse goals to map boxes to goal locations
        self.goal_locations = {}
        # The goals attribute in the Task class is a frozenset of goal facts.
        for goal_fact in self.goals:
            parts = get_parts(goal_fact)
            # Goal facts are typically (at ?box ?location)
            if parts[0] == 'at' and len(parts) == 3:
                box_name = parts[1]
                goal_loc_name = parts[2]
                self.goal_locations[box_name] = goal_loc_name

        # 2. Build the location graph from adjacent facts
        self.adj_list = {}
        locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'adjacent' and len(parts) == 4:
                loc1, loc2, direction = parts[1], parts[2], parts[3]
                locations.add(loc1)
                locations.add(loc2)
                # Add bidirectional edges assuming movement is possible both ways
                # if adjacent facts exist for both directions.
                # The BFS will find the shortest path regardless of direction used in PDDL fact.
                self.adj_list.setdefault(loc1, set()).add(loc2)
                self.adj_list.setdefault(loc2, set()).add(loc1) # Add reverse edge

        self.locations = list(locations) # List of all unique locations

        # 3. Precompute all-pairs shortest paths using BFS
        self.distances = {}
        all_locations = self.locations # Use the list of all locations

        for start_loc in all_locations:
            self.distances[start_loc] = {}
            # Initialize distances to infinity for all locations from start_loc
            for loc in all_locations:
                 self.distances[start_loc][loc] = float('inf')
            self.distances[start_loc][start_loc] = 0 # Distance from start to itself is 0

            # BFS queue
            q = deque([(start_loc, 0)])

            while q:
                current_loc, dist = q.popleft()

                # Get neighbors, handle locations with no adjacent facts (shouldn't happen for locations in the graph)
                neighbors = self.adj_list.get(current_loc, set())

                for neighbor in neighbors:
                    # If we found a shorter path (which is always the case the first time in BFS)
                    if self.distances[start_loc][neighbor] == float('inf'):
                        self.distances[start_loc][neighbor] = dist + 1
                        q.append((neighbor, dist + 1))

        # Define a large penalty for unreachable goals
        # Max possible distance in a connected graph is at most |V|-1.
        # Sum of distances for all boxes could be up to |Boxes| * (|V|-1).
        # Penalty should be larger than any possible sum of reachable distances.
        max_sum_reachable_dist = len(self.goal_locations) * len(self.locations) if self.locations else 0
        self.unreachable_penalty = max_sum_reachable_dist + 1 # Ensure it's larger


    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        based on the sum of box-goal distances.
        """
        state = node.state

        # Find current box locations
        current_box_locations = {}
        # Robot location is not used in this simple heuristic
        # robot_location = None

        for fact in state:
            parts = get_parts(fact)
            # Check if the fact is an 'at' predicate with 3 parts (predicate, obj, loc)
            # This distinguishes box locations (at box loc) from robot location (at-robot loc)
            if len(parts) == 3 and parts[0] == 'at':
                 box_name = parts[1]
                 loc_name = parts[2]
                 current_box_locations[box_name] = loc_name
            # elif len(parts) == 2 and parts[0] == 'at-robot':
            #      robot_location = parts[1]

        total_heuristic = 0

        # Sum distances for boxes not at their goals
        for box, goal_loc in self.goal_locations.items():
            current_loc = current_box_locations.get(box)

            # If a box is not found in the state, something is wrong, penalize heavily.
            # This shouldn't happen in a valid state generated by the planner.
            if current_loc is None:
                 return self.unreachable_penalty # Or a different large value

            if current_loc != goal_loc:
                # Get the precomputed distance from current_loc to goal_loc.
                # Use .get with a default of float('inf') to handle cases where
                # start_loc or goal_loc might not be in the precomputed table
                # (e.g., if the graph is disconnected, though unlikely in Sokoban instances).
                # The BFS precomputation should cover all locations found in adjacent facts.
                # If a location exists in the state but not in adjacent facts, it's an isolated node.
                # The BFS precomputation handles this by initializing distances to inf.
                dist = self.distances.get(current_loc, {}).get(goal_loc, float('inf'))

                if dist == float('inf'):
                    # Goal is unreachable from the box's current location.
                    # This state represents a potential deadlock or unsolvable situation.
                    # Return a large penalty.
                    return self.unreachable_penalty

                total_heuristic += dist

        # The heuristic is 0 if and only if all boxes are at their goal locations.
        return total_heuristic
