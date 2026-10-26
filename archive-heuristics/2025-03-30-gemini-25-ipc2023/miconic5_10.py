import math
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic # Assuming this path is correct based on examples
from collections import deque

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'miconic'.

    # Summary
    This heuristic estimates the number of actions (board, depart, up, down)
    required to reach a goal state where all specified passengers are served.
    The estimate is based on the number of passenger actions (boarding and
    departing) needed and an estimate of the lift movement required. It counts
    one 'board' action for each passenger currently waiting at their origin,
    one 'depart' action for each passenger not yet served. The lift movement
    cost is estimated based on the distance the lift needs to travel to reach
    the closest required floor (origin or destination) and the total range
    of floors that need to be visited.

    # Assumptions
    - Floors are arranged linearly, and the 'above' predicate defines direct
      adjacency between floors (e.g., f2 is directly above f1).
    - Floor indices can be consistently assigned based on the 'above' relations,
      starting from index 1 for the bottom floor.
    - The distance between two floors is the absolute difference of their indices.
    - The lift movement cost can be reasonably estimated by the sum of:
        1. The distance from the current lift position to the nearest floor
           that needs to be visited (for pickup or drop-off).
        2. The distance between the lowest and highest floors that need to be
           visited (the "span" of required travel).
    - The heuristic does not need to be admissible.

    # Heuristic Initialization
    - The constructor (`__init__`) parses the task's static information and goals.
    - It identifies all passengers and floors involved in the problem from static
      facts and the initial state.
    - It extracts the destination floor for each passenger from `(destin p f)` facts.
    - It builds a representation of the floor structure using `(above f1 f2)` facts.
    - It computes a numerical index for each floor based on its position in the
      linear arrangement, determined by traversing the 'above' relations from
      the bottom floor upwards using BFS. This allows calculating distances.
      Warnings are printed if the floor structure seems inconsistent.
    - It identifies the set of passengers that must be served to reach the goal state
      from `(served p)` goal facts.

    # Step-By-Step Thinking for Computing Heuristic
    The `__call__` method computes the heuristic value for a given state node:
    1. Check Goal Achievement: If all goal passengers are already served in the
       current state, the heuristic value is 0.
    2. Parse State: Identify the current floor of the lift (`lift-at`),
       passengers currently waiting at their origin (`origin`), passengers
       currently inside the lift (`boarded`), and passengers already served (`served`).
       Only consider passengers relevant to the goal.
    3. Calculate Unserved Passengers: Determine the set of passengers that are
       part of the goal but not yet served (`unserved_passengers = goal_passengers - served_passengers`).
    4. Estimate Passenger Action Cost:
       - Add 1 to the heuristic for each waiting passenger (estimated cost for `board`).
       - Add 1 to the heuristic for each unserved passenger (estimated cost for `depart`).
    5. Identify Required Floors: Determine the set of all floors the lift must
       visit (`FloorsToVisit`). This includes:
       - The origin floors of all waiting passengers.
       - The destination floors of all boarded passengers.
       - The destination floors of all waiting passengers who are also unserved.
    6. Estimate Lift Movement Cost:
       - If `FloorsToVisit` is empty, the movement cost is 0.
       - Otherwise:
         a. Get the precomputed index of the current lift floor (`current_lift_idx`).
         b. Get the precomputed indices of all floors in `FloorsToVisit`. Handle potential
            missing indices by using a fallback cost.
         c. Find the minimum (`min_target_idx`) and maximum (`max_target_idx`)
            indices among the target floors.
         d. Calculate the distance from the lift's current position to the
            closest target floor:
            `closest_target_dist = min(abs(current_lift_idx - idx))` for all `idx` in target indices.
         e. Calculate the span of the required travel:
            `span = max_target_idx - min_target_idx`.
         f. The estimated movement cost is `movement_cost = closest_target_dist + span`.
    7. Add Movement Cost: Add the estimated `movement_cost` to the heuristic value.
    8. Return Total Heuristic Value: The sum of passenger action costs and lift
       movement cost. Ensure the value is non-negative.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing static information and goals,
        and precomputing floor indices and distances.
        """
        self.task = task
        self.goals = task.goals
        self.static = task.static

        # Precompute data structures
        self._parse_static_info()
        self._compute_floor_indices()
        self._parse_goals()

    def _get_parts(self, fact):
        """Utility to parse PDDL fact string '(pred obj1 obj2)' into a list."""
        # Removes parentheses and splits by space
        return fact[1:-1].split()

    def _match(self, fact, *pattern):
        """Utility to check if a PDDL fact string matches a given pattern."""
        parts = self._get_parts(fact)
        if len(parts) != len(pattern):
            return False
        # Use fnmatch for potential wildcard matching if needed
        return all(fnmatch(part, pat) for part, pat in zip(parts, pattern))

    def _parse_static_info(self):
        """Extracts destinations, above relations, floors, and passengers."""
        self.destin = {}
        self.above_relations = set()
        floors_from_static = set()
        passengers_from_static = set()

        # Extract info from static facts
        for fact in self.static:
            parts = self._get_parts(fact)
            pred = parts[0]
            if pred == "destin":
                p, f = parts[1], parts[2]
                self.destin[p] = f
                passengers_from_static.add(p)
                floors_from_static.add(f)
            elif pred == "above":
                f1, f2 = parts[1], parts[2]
                floors_from_static.add(f1)
                floors_from_static.add(f2)
                self.above_relations.add((f1, f2))

        # Infer objects also from initial state if not in static
        floors_from_init = set()
        passengers_from_init = set()
        for fact in self.task.initial_state:
             parts = self._get_parts(fact)
             pred = parts[0]
             if pred == "lift-at":
                 floors_from_init.add(parts[1])
             elif pred == "origin":
                 passengers_from_init.add(parts[1])
                 floors_from_init.add(parts[2])

        # Combine objects from static and init
        self.floors = floors_from_static.union(floors_from_init)
        self.passengers = passengers_from_static.union(passengers_from_init)

        # Ensure all floors mentioned in relations/destinations are included
        for f1, f2 in self.above_relations:
            self.floors.add(f1)
            self.floors.add(f2)
        for f in self.destin.values():
            self.floors.add(f)
        # Ensure passengers mentioned in destin are included
        for p in self.destin.keys():
            self.passengers.add(p)


    def _compute_floor_indices(self):
        """
        Computes a numerical index for each floor based on 'above' relations.
        Assumes a linear structure where index increases with height.
        Uses BFS starting from the bottom floor(s).
        """
        self.floor_indices = {}
        if not self.floors:
            return

        adj = {f: set() for f in self.floors} # Stores floors directly below f
        rev_adj = {f: set() for f in self.floors} # Stores floors directly above f
        for f1, f2 in self.above_relations:
            # f1 is above f2
            adj[f1].add(f2)
            rev_adj[f2].add(f1)

        # Find bottom floor(s) - those with no floors below them
        bottom_floors = {f for f in self.floors if not adj[f]}

        if not bottom_floors:
             if len(self.floors) == 1:
                 # Handle single floor case
                 self.floor_indices = {list(self.floors)[0]: 1}
                 return
             else:
                # Problematic case: no bottom floor found (cycle? disconnected?)
                print(f"Warning: Could not find any bottom floor for {self.floors}. Assigning index 1 to all as fallback.")
                self.floor_indices = {f: 1 for f in self.floors}
                return

        # Initialize indices and BFS queue
        queue = deque()
        visited_for_bfs = set() # Tracks nodes added to queue

        # Start BFS from all bottom floors, assigning index 1
        for bf in bottom_floors:
            if bf not in visited_for_bfs:
                self.floor_indices[bf] = 1
                queue.append(bf)
                visited_for_bfs.add(bf)

        # Perform BFS to assign levels (indices)
        while queue:
            current_floor = queue.popleft()
            current_index = self.floor_indices[current_floor]

            # Explore floors directly above the current one
            for neighbor in rev_adj[current_floor]:
                # Calculate the correct index based on floors below the neighbor
                max_below_index = 0
                all_below_indexed = True
                if not adj[neighbor]: # Should have at least current_floor below it
                    all_below_indexed = False
                else:
                    for below_floor in adj[neighbor]:
                        if below_floor not in self.floor_indices:
                            all_below_indexed = False
                            break
                        max_below_index = max(max_below_index, self.floor_indices[below_floor])

                if all_below_indexed:
                    neighbor_index = max_below_index + 1
                    # Update index if this path gives a higher value (longest path from bottom)
                    if neighbor not in self.floor_indices or neighbor_index > self.floor_indices[neighbor]:
                         self.floor_indices[neighbor] = neighbor_index

                    # Add neighbor to queue only if it hasn't been queued before
                    if neighbor not in visited_for_bfs:
                         queue.append(neighbor)
                         visited_for_bfs.add(neighbor)

        # Final check: Ensure all floors got an index
        if len(self.floor_indices) != len(self.floors):
            unreached = self.floors - set(self.floor_indices.keys())
            print(f"Warning: Floor indexing incomplete. Total: {len(self.floors)}, Indexed: {len(self.floor_indices)}. Unreached: {unreached}. Assigning fallback index.")
            # Assign a fallback index (e.g., max index + 1) to unreached floors
            max_assigned_index = max(self.floor_indices.values()) if self.floor_indices else 0
            for f in unreached:
                self.floor_indices[f] = max_assigned_index + 1

    def _get_floor_distance(self, f1, f2):
        """Calculates distance between floors using precomputed indices."""
        # Handle cases where floors might be missing indices due to warnings
        idx1 = self.floor_indices.get(f1)
        idx2 = self.floor_indices.get(f2)
        if idx1 is None or idx2 is None:
             print(f"Warning: Cannot compute distance between {f1} and {f2}, index missing. Returning default distance 1.")
             return 1 # Minimal penalty distance
        return abs(idx1 - idx2)

    def _parse_goals(self):
        """Extracts the set of passengers that need to be served from goal facts."""
        self.goal_passengers = set()
        for goal in self.goals:
            # Example goal: "(served p1)"
            if self._match(goal, "served", "*"):
                passenger = self._get_parts(goal)[1]
                self.goal_passengers.add(passenger)
                # Ensure goal passengers are known, even if not in static/init
                self.passengers.add(passenger)


    def __call__(self, node):
        """
        Computes the heuristic value for the given state node.
        Estimates actions = (boards needed) + (departs needed) + (lift moves needed).
        """
        state = node.state

        # --- 1. Check Goal Achievement ---
        served_in_state = set()
        for fact in state:
             if self._match(fact, "served", "*"):
                 served_in_state.add(self._get_parts(fact)[1])

        if self.goal_passengers.issubset(served_in_state):
            # All required passengers are served
            return 0

        h = 0 # Initialize heuristic value

        # --- 2. Parse State ---
        current_lift_floor = None
        waiting_passengers = {} # passenger -> origin_floor
        boarded_passengers = set()

        for fact in state:
            parts = self._get_parts(fact)
            pred = parts[0]
            if pred == "lift-at":
                current_lift_floor = parts[1]
            elif pred == "origin":
                passenger = parts[1]
                # Only track passengers relevant to the goal
                if passenger in self.goal_passengers:
                    waiting_passengers[passenger] = parts[2]
            elif pred == "boarded":
                passenger = parts[1]
                # Only track passengers relevant to the goal
                if passenger in self.goal_passengers:
                    boarded_passengers.add(passenger)
            # served_in_state collected earlier

        if current_lift_floor is None:
            # Should not happen in valid states unless lift is removed?
            print("Warning: Lift location ('lift-at') not found in state. Returning high heuristic value.")
            return 1_000_000 # Return a large value to avoid exploring this path

        # --- 3. Calculate Unserved Passengers ---
        unserved_passengers = self.goal_passengers - served_in_state

        # --- 4. Estimate Passenger Action Cost ---
        # Cost for 'board' actions: one for each relevant waiting passenger
        h += len(waiting_passengers)
        # Cost for 'depart' actions: one for each relevant passenger not yet served
        h += len(unserved_passengers)

        # --- 5. Identify Required Floors ---
        origins_to_visit = set(waiting_passengers.values())
        destinations_to_visit = set()

        # Destinations for boarded passengers still needing service
        for p in boarded_passengers:
             if p in unserved_passengers: # Check if they still need serving (redundant?)
                 if p in self.destin:
                     destinations_to_visit.add(self.destin[p])
                 else:
                     print(f"Warning: Destination for boarded passenger {p} not found in static facts.")

        # Destinations for waiting passengers who still need service
        for p in waiting_passengers:
            if p in unserved_passengers: # Ensure they still need serving
                 if p in self.destin:
                     destinations_to_visit.add(self.destin[p])
                 else:
                     print(f"Warning: Destination for waiting passenger {p} not found in static facts.")

        floors_to_visit = origins_to_visit.union(destinations_to_visit)

        # --- 6. Estimate Lift Movement Cost ---
        movement_cost = 0
        if floors_to_visit:
            try:
                current_lift_idx = self.floor_indices.get(current_lift_floor)
                if current_lift_idx is None:
                    # Index missing for current floor
                    raise KeyError(f"Index for current lift floor '{current_lift_floor}' not found.")

                target_indices = set()
                for f in floors_to_visit:
                    idx = self.floor_indices.get(f)
                    if idx is None:
                        # Index missing for a target floor
                        raise KeyError(f"Index for target floor '{f}' not found.")
                    target_indices.add(idx)

                if not target_indices:
                     # This case should be covered by 'if floors_to_visit', but check again
                     movement_cost = 0
                else:
                    min_target_idx = min(target_indices)
                    max_target_idx = max(target_indices)

                    # Distance to the closest target floor index
                    closest_target_dist = min(abs(current_lift_idx - idx) for idx in target_indices)

                    # Span of the target floor indices
                    span = max_target_idx - min_target_idx

                    # Estimated movement cost
                    movement_cost = closest_target_dist + span

            except KeyError as e:
                 # Fallback if index calculation failed or state has unknown floors
                 print(f"Warning: KeyError during movement cost calculation: {e}. Using fallback cost.")
                 # Fallback: count number of distinct floors to visit as cost estimate
                 movement_cost = len(floors_to_visit)

        # --- 7. Add Movement Cost ---
        h += movement_cost

        # --- 8. Return Total Heuristic Value ---
        # Ensure heuristic is non-negative
        return max(0, h)
