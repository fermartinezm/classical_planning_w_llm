import collections
import itertools
from fnmatch import fnmatch
# Assuming the Heuristic base class is accessible via this path
# Adjust the import path if necessary based on the project structure
from heuristics.heuristic_base import Heuristic

# Helper function to parse PDDL facts like "(predicate arg1 arg2)"
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Removes parentheses and splits by space.
    Example: "(lift-at f2)" -> ["lift-at", "f2"]
    """
    # Ensure fact is a string and has parentheses before stripping
    if isinstance(fact, str) and len(fact) > 1 and fact.startswith('(') and fact.endswith(')'):
        return fact[1:-1].split()
    # Return an empty list or raise error if format is unexpected
    return []

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic (elevator) domain.

    # Summary
    This heuristic estimates the number of actions required to serve all passengers
    whose goal is to be in the 'served' state. It calculates the estimated cost
    for each unserved passenger individually and sums these costs. The cost for a
    passenger includes the estimated lift movements needed to pick them up (if waiting)
    and drop them off at their destination, plus the necessary board and depart actions.

    # Assumptions
    - The `above` predicate defines direct adjacency between floors (i.e., floors
      that the lift can move between in a single 'up' or 'down' action). The domain
      actions `up` and `down` confirm this interpretation.
    - The cost of moving between adjacent floors is 1 (one `up` or `down` action).
    - The heuristic sums costs independently for each passenger. This is a form of
      relaxation and might overestimate the total actions required, as the lift can
      potentially serve multiple passengers during the same trip segments. This
      overestimation is acceptable for a non-admissible heuristic aimed at guiding
      a greedy best-first search effectively.
    - All passengers specified in the '(served p)' goal conditions have a
      corresponding '(destin p f)' fact defined in the static information.
    - The floor structure forms a single connected component reachable by the lift;
      otherwise, distances might be infinite for some passengers, leading to an
      infinite heuristic value if they need service between disconnected parts.

    # Heuristic Initialization
    - Extracts all unique floor objects mentioned in the task (static, init, goal).
    - Parses static `(above f1 f2)` facts to build an undirected graph representing
      floor adjacency. An edge exists between f1 and f2 if `(above f1 f2)` or
      `(above f2 f1)` is present (implicitly, as `up`/`down` actions check `above`
      in one direction).
    - Precomputes the shortest path distances (number of 'up'/'down' actions)
      between all pairs of floors using Breadth-First Search (BFS) on the floor graph.
      Stores these distances in a dictionary `self.distances[(floor1, floor2)]`.
      Unreachable pairs have a distance of infinity.
    - Parses static `(destin p f)` facts to store the destination floor for each
      passenger in `self.destinations`.
    - Identifies all passengers `p` for whom `(served p)` is a goal condition. Stores
      these passengers in `self.goal_passengers`.

    # Step-By-Step Thinking for Computing Heuristic
    1.  Check if the current `node.state` satisfies all goal conditions using
        `self.task.goal_reached(node.state)`. If yes, return 0 immediately.
    2.  Initialize the total heuristic estimate `h = 0`.
    3.  Parse the current `state` to find the current floor of the lift `lift_floor`
        (from the single `(lift-at ?f)` fact). If not found, return infinity (invalid state).
    4.  Parse the current `state` to determine the status of all passengers:
        - Create a set `boarded_passengers` containing passengers `p` for whom `(boarded p)` is true.
        - Create a dictionary `waiting_passengers` mapping passenger `p` to their origin floor `f`
          for whom `(origin p f)` is true.
        - Create a set `served_passengers` containing passengers `p` for whom `(served p)` is true.
    5.  Iterate through each passenger `p` in `self.goal_passengers` (those required to be served).
    6.  For each such passenger `p`:
        a. If `p` is already in `served_passengers`, their goal is met, so they contribute 0
           to the heuristic. Continue to the next passenger.
        b. Retrieve the destination floor `dest_f` for passenger `p` from the
           precomputed `self.destinations` map. If `p` has no destination defined (which
           should not happen for valid problems where `p` is a goal passenger), return infinity.
        c. **Case 1: Passenger `p` is boarded (`p` in `boarded_passengers`).**
           - Calculate the distance from the current lift floor `lift_f` to `dest_f`:
             `dist = self.distances.get((lift_f, dest_f), float('inf'))`.
           - If `dist` is infinity, the destination is unreachable; return `float('inf')`.
           - Add `dist + 1` to `h`. This accounts for the lift movement actions (`dist`)
             plus one `depart` action.
        d. **Case 2: Passenger `p` is waiting (`p` in `waiting_passengers`).**
           - Get the origin floor `origin_f = waiting_passengers[p]`.
           - Calculate distance from `lift_f` to `origin_f`:
             `dist1 = self.distances.get((lift_f, origin_f), float('inf'))`.
           - Calculate distance from `origin_f` to `dest_f`:
             `dist2 = self.distances.get((origin_f, dest_f), float('inf'))`.
           - If either `dist1` or `dist2` is infinity, return `float('inf')`.
           - Add `dist1 + 1 + dist2 + 1` to `h`. This accounts for movement to origin (`dist1`),
             one `board` action (+1), movement to destination (`dist2`), and one `depart` action (+1).
        e. **Case 3: Passenger `p` is not served, not boarded, and not waiting.**
           - This state is unusual for a passenger required in the goal under normal
             domain semantics. It might imply the passenger started at their destination
             floor but the `(origin p f)` fact was true initially and has been removed
             (e.g., by boarding) but the `(boarded p)` fact isn't present yet, or some
             other inconsistency.
           - A simple check: if the passenger initially started at their destination floor
             (i.e., `(origin p f)` and `(destin p f)` referred to the same floor `f` in the
             initial state), they still need a lift cycle (move-board-depart). We estimate
             this cost.
           - If the passenger didn't start at their destination, this state is considered
             problematic or unable to reach the goal for this passenger, so return `float('inf')`.
    7.  After iterating through all goal passengers, if the calculated heuristic `h` is 0
        (and we already know it's not a goal state from step 1), it means there were no
        unserved goal passengers requiring action (e.g., all necessary moves had zero distance,
        which is unlikely if actions are needed). Return 1 to ensure the heuristic is
        strictly positive for non-goal states.
    8.  Otherwise, return the calculated total cost `h`.
    """

    def __init__(self, task):
        super().__init__(task)
        self.task = task # Store task for access to initial state and goal check
        static_facts = task.static

        # --- Extract Floors ---
        self.floors = set()
        # Combine all known facts to find all floor objects
        all_facts = task.initial_state.union(task.goals).union(static_facts)
        for fact in all_facts:
            parts = get_parts(fact)
            if not parts: continue # Skip if parsing failed

            # Add objects identified as floors based on predicates that use them
            if parts[0] == 'above':
                # Arguments are floors
                self.floors.add(parts[1])
                self.floors.add(parts[2])
            elif parts[0] in ['lift-at', 'origin', 'destin']:
                # Assume the last argument is the floor
                self.floors.add(parts[-1])

        # --- Build Adjacency Graph ---
        # Adjacency represents floors the lift can move between in one step
        adj = collections.defaultdict(set)
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            if parts[0] == 'above':
                # (above f1 f2) means f1 is directly above f2.
                # Action 'up' moves from f1 to f2 if (above f1 f2).
                # Action 'down' moves from f1 to f2 if (above f2 f1).
                # So, an edge exists if either (above f1 f2) or (above f2 f1) holds.
                f1, f2 = parts[1], parts[2]
                adj[f1].add(f2)
                adj[f2].add(f1) # Undirected edge for movement capability

        # --- Precompute All-Pairs Shortest Paths (Distances) using BFS ---
        self.distances = collections.defaultdict(lambda: float('inf'))
        for start_node in self.floors:
            # Skip potential empty strings or invalid floor names
            if not start_node: continue
            self.distances[(start_node, start_node)] = 0
            queue = collections.deque([(start_node, 0)])
            # Keep track of visited nodes for this specific BFS run
            visited_bfs = {start_node: 0}

            while queue:
                current_node, dist = queue.popleft()

                # Explore neighbors based on the adjacency graph
                for neighbor in adj.get(current_node, set()):
                    if neighbor not in visited_bfs:
                        visited_bfs[neighbor] = dist + 1
                        self.distances[(start_node, neighbor)] = dist + 1
                        queue.append((neighbor, dist + 1))
                    # Optimization: If we find a shorter path (shouldn't happen with BFS)
                    # elif dist + 1 < visited_bfs[neighbor]:
                    #     visited_bfs[neighbor] = dist + 1
                    #     self.distances[(start_node, neighbor)] = dist + 1
                    #     # Re-add to queue if needed, though BFS guarantees shortest path first time

        # --- Store Passenger Destinations ---
        self.destinations = {}
        for fact in static_facts:
             parts = get_parts(fact)
             if not parts: continue
             if parts[0] == 'destin':
                 # (destin passenger floor)
                 passenger, floor = parts[1], parts[2]
                 self.destinations[passenger] = floor

        # --- Identify Goal Passengers ---
        self.goal_passengers = set()
        for goal in self.task.goals:
             parts = get_parts(goal)
             if not parts: continue
             # Assume goals are primarily (served passenger)
             if parts[0] == 'served':
                 passenger = parts[1]
                 self.goal_passengers.add(passenger)
                 # Sanity check: ensure goal passengers have destinations defined
                 if passenger not in self.destinations:
                     # This indicates a potentially ill-formed problem instance
                     print(f"ERROR: Heuristic init: Passenger {passenger} required by goal "
                           f"but has no destination fact in static info.")
                     # Consider raising an error or handling appropriately

        # Store initial origins for the edge case check later
        self._initial_origins = {}
        for fact in self.task.initial_state:
            parts = get_parts(fact)
            if not parts: continue
            if parts[0] == 'origin':
                self._initial_origins[parts[1]] = parts[2]


    def __call__(self, node):
        state = node.state

        # --- Check if Goal State ---
        # Use the task's method to check if the goal is reached
        if self.task.goal_reached(state):
            return 0

        h = 0

        # --- Get Current State Information ---
        lift_floor = None
        boarded_passengers = set()
        waiting_passengers = {} # passenger -> origin_floor
        served_passengers = set()

        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip malformed facts

            predicate = parts[0]
            if predicate == 'lift-at':
                lift_floor = parts[1]
            elif predicate == 'boarded':
                boarded_passengers.add(parts[1])
            elif predicate == 'origin':
                # Passenger p is waiting at floor f
                waiting_passengers[parts[1]] = parts[2]
            elif predicate == 'served':
                served_passengers.add(parts[1])

        # Lift location is crucial for calculating move costs
        if lift_floor is None:
             print("ERROR: Heuristic call: Lift location not found in state.")
             return float('inf') # Cannot compute heuristic without lift location

        # --- Calculate Heuristic Cost ---
        for p in self.goal_passengers:
            # Skip passengers already served
            if p in served_passengers:
                continue

            dest_floor = self.destinations.get(p)
            # If a goal passenger has no destination, the problem is likely unsolvable
            if dest_floor is None:
                 print(f"ERROR: Heuristic call: Passenger {p} needed for goal has no destination.")
                 return float('inf')

            # Case 1: Passenger is boarded
            if p in boarded_passengers:
                # Cost = move lift from current floor to destination + 1 depart action
                move_cost = self.distances.get((lift_floor, dest_floor), float('inf'))
                # If destination is unreachable, goal cannot be met
                if move_cost == float('inf'):
                    # print(f"Debug: Passenger {p} boarded, dest {dest_floor} unreachable from {lift_floor}")
                    return float('inf')
                h += move_cost + 1

            # Case 2: Passenger is waiting at an origin floor
            elif p in waiting_passengers:
                origin_floor = waiting_passengers[p]
                # Cost = move lift to origin + 1 board + move lift from origin to dest + 1 depart
                move1_cost = self.distances.get((lift_floor, origin_floor), float('inf'))
                move2_cost = self.distances.get((origin_floor, dest_floor), float('inf'))

                # If either origin or destination is unreachable, goal cannot be met
                if move1_cost == float('inf') or move2_cost == float('inf'):
                    # print(f"Debug: Passenger {p} waiting at {origin_floor}, "
                    #       f"origin reachable: {move1_cost != float('inf')}, "
                    #       f"dest {dest_floor} reachable from origin: {move2_cost != float('inf')}")
                    return float('inf')
                h += move1_cost + 1 + move2_cost + 1

            # Case 3: Passenger is not served, not boarded, and not waiting
            else:
                # This state is unusual for a required passenger.
                # Check if they started at their destination (edge case).
                initial_origin = self._initial_origins.get(p)
                if initial_origin is not None and initial_origin == dest_floor:
                    # Passenger started at destination, needs a pickup/dropoff cycle.
                    # Cost: move lift to origin(=dest) + board + depart
                    move_to_origin_cost = self.distances.get((lift_floor, initial_origin), float('inf'))
                    if move_to_origin_cost == float('inf'):
                        # print(f"Debug: Passenger {p} started at dest {initial_origin}, but unreachable from {lift_floor}")
                        return float('inf')
                    # Cost assumes 0 move from origin to dest since they are the same
                    h += move_to_origin_cost + 1 + 1
                else:
                    # If not the edge case, this state seems invalid or problematic for reaching the goal for p.
                    # print(f"Warning: Heuristic call: Passenger {p} required for goal is in unexpected state "
                    #       f"(not served, boarded, or waiting, and didn't start at destination {dest_floor}).")
                    return float('inf') # Assume goal is unreachable from this state for p

        # --- Final Adjustment ---
        # The first check `if self.task.goal_reached(state): return 0` handles goal states.
        # If h is 0 after calculations, but it's not a goal state, it implies no actions
        # were estimated as needed (e.g., no unserved passengers, or all moves were 0 cost).
        # Return 1 to ensure the heuristic is > 0 for non-goal states.
        if h == 0:
            # We know it's not a goal state because of the check at the start.
            return 1

        # Return the calculated heuristic value
        return h
