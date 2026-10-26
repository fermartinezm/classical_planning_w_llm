import sys
from fnmatch import fnmatch
from collections import deque # Used for BFS in floor level calculation

# Try to import the base class. If not available, define a dummy placeholder.
# This allows the code to be syntactically correct for analysis, but requires
# the actual base class to be available at runtime.
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    print("Warning: Heuristic base class not found. Defining a dummy class.", file=sys.stderr)
    class Heuristic:
        def __init__(self, task):
            """Dummy initializer."""
            pass
        def __call__(self, node):
            """Dummy call method."""
            raise NotImplementedError("Heuristic base class not found.")

# Helper functions
def get_parts(fact):
    """
    Extracts predicate and arguments from a PDDL fact string.
    Example: "(pred obj1 obj2)" -> ["pred", "obj1", "obj2"]
    Handles potential surrounding whitespace and empty facts.
    """
    stripped_fact = fact.strip()
    if len(stripped_fact) < 2 or stripped_fact[0] != '(' or stripped_fact[-1] != ')':
        # Return empty list for invalid format or empty fact "()"
        return []
    return stripped_fact[1:-1].split()

def match(fact_parts, *pattern):
    """
    Checks if the parts of a fact match a given pattern.
    Supports '*' wildcard in the pattern using fnmatch.
    Example: match(["at", "p1", "f1"], "at", "*", "f1") -> True
    """
    if len(fact_parts) != len(pattern):
        return False
    # fnmatch provides wildcard matching (*, ?, [])
    return all(fnmatch(part, pat) for part, pat in zip(fact_parts, pattern))

class miconicHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic elevator domain.

    # Summary
    Estimates the cost to reach the goal state (all passengers served) by summing:
    1. The number of required board and depart actions for unserved passengers.
    2. An estimate of the lift's travel distance required to visit all necessary floors
       (origins of waiting passengers and destinations of waiting/boarded passengers).

    # Assumptions
    - Floors are linearly ordered (e.g., f1 above f2 above f3...).
    - The `above` predicate correctly defines this order, implying a single path
      from top to bottom. A floor `f1` is considered "higher" than `f2` if `(above f1 f2)` exists.
    - Floor names allow determining the order (e.g., 'f1', 'f2', ...) as a
      fallback if `above` facts are insufficient or ambiguous. Level 1 is assigned
      to the highest floor based on sorting or `above` facts.
    - The lift moves between adjacent floors using 'up' or 'down' actions,
      each costing 1. Distance is the number of steps (difference in levels).
    - The heuristic is designed for greedy best-first search and is not
      necessarily admissible.

    # Heuristic Initialization
    - Parses static facts (`task.static`) to store passenger destinations (`destin`).
    - Determines the level (numerical representation, e.g., 1 for highest floor)
      of each floor based on the `above` facts. It uses a Breadth-First Search (BFS)
      starting from the top floor(s) (those with no floor above them). If this fails
      or finds inconsistencies, it falls back to sorting floor names numerically
      (assuming 'fX' format where X is an integer).
    - Stores the set of all passenger names defined by `destin` facts.

    # Step-By-Step Thinking for Computing Heuristic
    1.  **Identify Current State:** From the input `node.state`:
        - Find the lift's current floor (`lift_floor`) using `(lift-at ?f)`.
        - Identify waiting passengers (`p` where `(origin p floor)` is true)
          and their origin floors. Store in `waiting` dict: `{p: origin_f}`.
        - Identify boarded passengers (`p` where `(boarded p)` is true). Store
          in `boarded` set: `{p1, p2, ...}`.
        - Identify served passengers (`p` where `(served p)` is true). Store
          in `served` set.
    2.  **Check Goal:** If the `served` set contains all passengers known from
        initialization (`self.passengers`), the goal is reached. Return 0.
    3.  **Calculate Action Costs:**
        - Initialize heuristic value `h = 0`.
        - Add 1 to `h` for each passenger in `waiting` (requires a `board` action).
        - Add 1 to `h` for each passenger in `waiting` (requires a `depart` action later).
        - Add 1 to `h` for each passenger in `boarded` (requires a `depart` action).
        - Total action cost = `2 * len(waiting) + len(boarded)`.
    4.  **Calculate Movement Costs:**
        - Determine the set of `required_floors` the lift must visit:
            - The origin floor (`origin_f`) for each passenger `p` in `waiting`.
            - The destination floor (`destin[p]`) for each passenger `p` in `waiting`.
            - The destination floor (`destin[p]`) for each passenger `p` in `boarded`.
        - If `required_floors` is empty (no passengers waiting or boarded),
          movement cost is 0.
        - Otherwise:
            - Find the minimum (`min_req_level`) and maximum (`max_req_level`)
              floor levels among the `required_floors`. Use `self.floor_levels`.
              Handle potential `KeyError` if a floor level is unknown (indicates an issue).
            - Get the lift's current floor level (`current_level`) using `self.floor_levels`.
              Handle potential `KeyError`.
            - Calculate the distance the lift needs to travel to reach the
              operational range [`min_req_level`, `max_req_level`]:
              `dist_to_range = max(0, current_level - max_req_level) + max(0, min_req_level - current_level)`
            - Estimate the travel distance within the range as the span:
              `span = max_req_level - min_req_level`.
            - Total estimated movement cost = `dist_to_range + span`. Add this to `h`.
            - If any floor level lookup failed during this process, add a small penalty (e.g., 1)
              to the movement cost instead of the calculated value, to avoid
              a potentially zero cost when movement is needed but cannot be accurately estimated.
    5.  **Final Value:**
        - Return the total sum `h`.
        - Ensure `h > 0` for non-goal states. If `h` is 0 but the state is not
          a goal state (e.g., due to errors or edge cases), return 1.
    """

    def __init__(self, task):
        super().__init__(task) # Initialize base class
        self.goals = task.goals
        static_facts = task.static

        self.destin = {} # passenger -> destination_floor
        self.floor_levels = {} # floor -> level (int, e.g., 1=highest)
        self.floors = set() # All floor objects encountered

        adj = {} # floor -> set of floors directly below (from 'above f1 f2', f1 -> f2)
        rev_adj = {} # floor -> set of floors directly above (from 'above f1 f2', f2 -> f1)

        # --- Step 1: Parse static facts and build floor graph ---
        for fact in static_facts:
            parts = get_parts(fact)
            if not parts: continue

            if match(parts, "destin", "?p", "?f"):
                passenger, floor = parts[1], parts[2]
                self.destin[passenger] = floor
                self.floors.add(floor) # Track floor
            elif match(parts, "above", "?f1", "?f2"):
                f1, f2 = parts[1], parts[2]
                self.floors.add(f1)
                self.floors.add(f2)
                adj.setdefault(f1, set()).add(f2)
                rev_adj.setdefault(f2, set()).add(f1)

        # Add floors from initial state just in case they weren't in static facts
        for fact in task.initial_state:
             parts = get_parts(fact)
             if not parts: continue
             # Check for origin or lift-at facts to ensure all floors are captured
             if match(parts, "origin", "?p", "?f"):
                 floor = parts[2]
                 self.floors.add(floor)
             elif match(parts, "lift-at", "?f"):
                 floor = parts[1]
                 self.floors.add(floor)

        # --- Step 2: Determine floor levels ---
        if not self.floors:
            print("Warning: No floors found in the task.", file=sys.stderr)
        elif len(self.floors) == 1:
            self.floor_levels[list(self.floors)[0]] = 1 # Assign level 1 to single floor
        else:
            # Try BFS from top floor(s) to establish levels
            # Top floors are those not present as f2 in (above f1 f2)
            top_floors = self.floors - set(rev_adj.keys())
            queue = deque() # Use deque for efficient BFS queue
            visited = set()
            levels_ok = True # Flag to track if BFS assignment is consistent

            if not top_floors:
                print("Warning: No top floor found (potential cycle or incomplete 'above' data). Attempting fallback.", file=sys.stderr)
                levels_ok = False
            else:
                # Initialize queue with top floors and level 1
                # Sort top floors for deterministic initialization
                for f in sorted(list(top_floors)):
                    if f not in self.floor_levels:
                        queue.append((f, 1))
                        visited.add(f)
                        self.floor_levels[f] = 1

                while queue:
                    f, lvl = queue.popleft() # BFS uses popleft

                    # Explore floors directly below f
                    if f in adj:
                        # Sort neighbors for deterministic traversal
                        for neighbor in sorted(list(adj[f])):
                            if neighbor not in visited:
                                visited.add(neighbor)
                                self.floor_levels[neighbor] = lvl + 1
                                queue.append((neighbor, lvl + 1))
                            elif self.floor_levels[neighbor] != lvl + 1:
                                # Inconsistency detected (e.g., non-linear structure like diamond)
                                # This heuristic assumes linearity, warn user.
                                print(f"Warning: Inconsistent level for floor {neighbor}. "
                                      f"Current: {self.floor_levels[neighbor]}, Proposed via {f}: {lvl + 1}. "
                                      "Keeping first assigned level. Heuristic might be inaccurate.", file=sys.stderr)
                                # Optionally set levels_ok = False if strict linearity is required

            # Check if BFS covered all floors
            if len(self.floor_levels) != len(self.floors):
                print(f"Warning: BFS did not cover all floors ({len(self.floor_levels)}/{len(self.floors)}). "
                      f"Uncovered: {self.floors - set(self.floor_levels.keys())}. Attempting fallback.", file=sys.stderr)
                levels_ok = False

            # Fallback: If BFS failed or was inconsistent, try sorting floor names
            if not levels_ok or len(self.floor_levels) != len(self.floors):
                print("Falling back to assigning floor levels based on numerical sort of names (e.g., f1, f2...).", file=sys.stderr)
                try:
                    # Assumes names like 'f1', 'f10', etc.
                    sorted_floors = sorted(list(self.floors), key=lambda f_name: int(f_name[1:]))
                    # Overwrite all levels based on sorted order
                    self.floor_levels = {f: i + 1 for i, f in enumerate(sorted_floors)}
                    print(f"Assigned levels via name sorting: {self.floor_levels}", file=sys.stderr)
                except (ValueError, IndexError):
                    # If name sorting fails, we cannot determine levels reliably
                    raise ValueError("FATAL: Cannot determine floor order. BFS failed and floor names are not in 'fX' format.")

        # --- Step 3: Store passenger list ---
        self.passengers = set(self.destin.keys())
        if not self.passengers:
             print("Warning: No passengers found with destinations ('destin' facts).", file=sys.stderr)

        # --- Step 4: Sanity check floor levels ---
        if self.floors and not self.floor_levels:
             raise ValueError("FATAL: Floor levels could not be determined after initialization.")
        # print(f"DEBUG: Initialized miconicHeuristic. Floor levels: {self.floor_levels}", file=sys.stderr)


    def _get_floor_level(self, floor):
        """Safely get the precomputed floor level. Raises KeyError if unknown."""
        if floor not in self.floor_levels:
            # This indicates an issue with initialization or an unexpected floor name in the state
            raise KeyError(f"Floor level for '{floor}' is unknown. Check PDDL files and heuristic initialization logic.")
        return self.floor_levels[floor]

    def __call__(self, node):
        """Calculates the heuristic value for the given state node."""
        state = node.state
        h = 0 # Initialize heuristic value

        lift_floor = None
        waiting = {} # passenger -> origin_floor
        boarded = set() # passengers currently boarded
        served = set() # passengers already served

        # --- Step 1: Parse current state ---
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty or invalid facts

            # Use match helper for clarity and robustness
            if match(parts, "lift-at", "?f"):
                lift_floor = parts[1]
            elif match(parts, "origin", "?p", "?f"):
                passenger = parts[1]
                # Only consider passengers we know about (have a destination)
                if passenger in self.passengers:
                    waiting[passenger] = parts[2]
            elif match(parts, "boarded", "?p"):
                 passenger = parts[1]
                 if passenger in self.passengers:
                    boarded.add(passenger)
            elif match(parts, "served", "?p"):
                 passenger = parts[1]
                 if passenger in self.passengers:
                    served.add(passenger)

        # Check for state validity regarding lift location
        if lift_floor is None:
            # This might happen in trivial problems or malformed states
            if not self.floors: return 0 # No floors, maybe goal state?
            print(f"Error: Lift location 'lift-at' not found in state. State: {state}", file=sys.stderr)
            # Return infinity for greedy search to avoid exploring invalid states
            return float('inf')

        # --- Step 2: Check Goal ---
        # Goal is reached if all passengers we know about are in the 'served' set
        if len(served) == len(self.passengers):
             # Sanity check: if all served, no known passengers should be waiting or boarded
             if __debug__: # Assertions might be disabled in optimized runs
                 assert all(p not in waiting for p in self.passengers), f"Goal state but passenger {next(p for p in self.passengers if p in waiting)} is waiting"
                 assert all(p not in boarded for p in self.passengers), f"Goal state but passenger {next(p for p in self.passengers if p in boarded)} is boarded"
             return 0 # Goal state reached

        # --- Step 3: Calculate Action Costs ---
        num_waiting = len(waiting)
        num_boarded = len(boarded)
        # Cost = board actions for waiting + depart actions for waiting + depart actions for boarded
        h += num_waiting + num_waiting + num_boarded

        # --- Step 4: Calculate Movement Costs ---
        required_floors = set()
        min_req_level = float('inf')
        max_req_level = float('-inf')
        movement_calculation_ok = True # Track if all needed levels can be found

        # Combine waiting and boarded passengers for easier iteration
        involved_passengers = list(waiting.items()) + [(p, None) for p in boarded] # (passenger, origin_floor or None)

        for p, origin_f in involved_passengers:
            try:
                destin_f = self.destin[p] # Get destination floor for passenger p
                destin_level = self._get_floor_level(destin_f)
                required_floors.add(destin_f)
                min_req_level = min(min_req_level, destin_level)
                max_req_level = max(max_req_level, destin_level)

                if origin_f: # If passenger is waiting (origin_f is not None)
                    origin_level = self._get_floor_level(origin_f)
                    required_floors.add(origin_f)
                    min_req_level = min(min_req_level, origin_level)
                    max_req_level = max(max_req_level, origin_level)

            except KeyError as e:
                 # Failed to get level for origin or destination of this passenger
                 print(f"Warning: Heuristic calculation error for passenger {p}: {e}. Movement cost might be inaccurate.", file=sys.stderr)
                 movement_calculation_ok = False
                 # Continue collecting other floors if possible, but mark calculation as potentially flawed

        movement_cost = 0
        if required_floors: # Only calculate if there are floors to visit
            if not movement_calculation_ok:
                # If any level lookup failed, use a default penalty for movement
                # Penalizing by 1 seems reasonable as a minimum estimate when details fail
                movement_cost = 1
                print(f"Using default movement penalty {movement_cost} due to level lookup errors.", file=sys.stderr)
            else:
                try:
                    current_level = self._get_floor_level(lift_floor)

                    # Ensure min/max levels are valid (should be if movement_calculation_ok is True)
                    if min_req_level == float('inf'):
                         # This case should ideally not happen if required_floors is non-empty and calculation was ok
                         print("Warning: min_req_level is inf despite required_floors being non-empty and calculation marked OK.", file=sys.stderr)
                         movement_cost = 1 # Default penalty
                    else:
                        # Distance to get into the required range of floors
                        dist_to_range = max(0, current_level - max_req_level) + max(0, min_req_level - current_level)

                        # Span of the required floors
                        span = max_req_level - min_req_level

                        movement_cost = dist_to_range + span
                except KeyError as e:
                     # Failed to get level for the lift's current floor
                     print(f"Warning: Heuristic calculation error: {e}. Using default movement penalty.", file=sys.stderr)
                     movement_cost = 1 # Default penalty

        h += movement_cost

        # --- Step 5: Final Value ---
        # Ensure heuristic is non-zero for non-goal states
        # This handles cases where actions=0, movement=0 but goal not met (e.g., error state, or calculation issue)
        if h == 0 and len(served) != len(self.passengers):
             # If no actions needed and no movement calculated, but not goal state -> return 1
             # This ensures the search progresses and doesn't falsely identify a non-goal as goal.
             # print(f"Warning: Heuristic calculated 0 for non-goal state. Returning 1. State: {state}", file=sys.stderr)
             return 1

        return h

