from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic
from collections import deque

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(at obj loc)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

def bfs(graph, start_node):
    """Perform BFS to find shortest distances from start_node to all reachable nodes."""
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = deque([start_node])
    while queue:
        current = queue.popleft()
        for neighbor in graph.get(current, []):
            if distances[neighbor] == float('inf'):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the minimum number of actions required to tighten all
    goal nuts. It considers the cost of moving the man to the nut location,
    finding and picking up a usable spanner (if needed), and performing the
    tighten action. It accounts for spanners becoming unusable after one use.

    # Assumptions
    - Nuts do not move from their initial locations.
    - Spanners become unusable after one tighten action.
    - The man can only carry one spanner at a time (implicitly by the predicate definition).
    - The cost of walking between linked locations is 1.
    - Problems are assumed solvable with the initial set of usable spanners and reachable locations.
      States leading to unsolvable situations (e.g., unreachable locations, no usable spanners left)
      are assigned a large penalty.

    # Heuristic Initialization
    - Extracts the goal conditions to identify which nuts need tightening.
    - Identifies the man object name.
    - Builds a graph of locations based on `link` facts from static information.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies the initial locations of all goal nuts (assuming they don't move).

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the cost by simulating a greedy process of tightening
    each loose goal nut one by one in a fixed order (alphabetical by nut name).
    For each loose goal nut `N` at location `L_N`:

    1.  **Move to Nut Location:** Calculate the shortest distance from the man's
        current estimated location (`current_man_loc`) to `L_N`. Add this distance
        to the total heuristic cost. Update `current_man_loc` to `L_N`. If `L_N`
        is unreachable, add a large penalty and stop.

    2.  **Get a Usable Spanner (if needed):** Check if the man is currently
        estimated to be carrying a usable spanner (`current_is_carrying_usable_spanner`).
        If not:
        a.  Find the nearest available usable spanner (`S`) on the ground from
            the `current_man_loc` (`L_N`). Available spanners are those that
            were initially usable and haven't been "used" by the heuristic yet.
        b.  If a usable spanner `S` is found at location `L_S` (and `L_S` is reachable):
            i.  Calculate the shortest distance from `current_man_loc` (`L_N`)
                to `L_S`. Add this distance to the cost. Update `current_man_loc`
                to `L_S`.
            ii. Add 1 for the `pickup_spanner` action.
            iii. Mark the man as carrying a usable spanner (`current_is_carrying_usable_spanner = True`).
            iv. Remove `S` from the pool of available usable spanners on the ground.
            v.  Calculate the shortest distance from the new `current_man_loc` (`L_S`)
                back to `L_N`. Add this distance to the cost. Update `current_man_loc`
                to `L_N`. If `L_N` is unreachable from `L_S`, add a large penalty and stop.
        c.  If no usable spanner is found among the available ones (and none is carried):
            This state is likely unsolvable with the remaining resources. Add a large
            penalty for this nut and all remaining nuts and stop processing further nuts.

    3.  **Tighten Nut:** If the man is estimated to be carrying a usable spanner,
        add 1 for the `tighten_nut` action. Mark the spanner as unusable by setting
        `current_is_carrying_usable_spanner = False`.

    The total heuristic value is the sum of costs accumulated for each nut.
    If the goal is already reached (all goal nuts are tightened), the set of
    loose goal nuts is empty, and the heuristic returns 0.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and computing distances."""
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        initial_state = task.initial_state # Initial state facts

        # Identify goal nuts
        self.goal_nuts = {
            args[0] for goal in self.goals
            if match(goal, "tightened", "*")
            for args in [get_parts(goal)[1:]]
        }

        # Identify the man object name
        self.man_name = None
        # Try finding man from initial carrying fact
        for fact in initial_state:
            if match(fact, "carrying", "*", "*"):
                self.man_name = get_parts(fact)[1]
                break
        # If not carrying initially, try finding man from goal carrying fact
        if self.man_name is None:
            for goal in self.goals:
                if match(goal, "carrying", "*", "*"):
                    self.man_name = get_parts(goal)[1]
                    break
        # If still not found, try finding man from initial 'at' fact
        # Assume the first object in an '(at obj loc)' fact that is NOT a spanner or nut is the man.
        if self.man_name is None:
            # Get names of spanners and nuts from initial state 'at' facts (case-insensitive)
            initial_locatables = {get_parts(f)[1] for f in initial_state if match(f, "at", "*", "*")}
            initial_spanners = {obj for obj in initial_locatables if 'spanner' in obj.lower()}
            initial_nuts = {obj for obj in initial_locatables if 'nut' in obj.lower()}

            for fact in initial_state:
                if match(fact, "at", "*", "*"):
                    obj, loc = get_parts(fact)[1:]
                    if obj not in initial_spanners and obj not in initial_nuts:
                         self.man_name = obj
                         break

        # If man_name is still None, fallback to 'bob' or raise error.
        # print(f"Debug: Identified man name: {self.man_name}")
        if self.man_name is None:
             # print("Warning: Could not identify man object. Assuming 'bob'.")
             self.man_name = 'bob' # Fallback - fragile

        # Map nuts to their initial locations (assuming nuts don't move)
        self.nut_locations = {}
        for fact in initial_state: # Nuts are initially placed
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # Check if this object is one of the goal nuts
                 if obj in self.goal_nuts:
                     self.nut_locations[obj] = loc

        # Build location graph from link facts
        self.location_graph = {}
        locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                locations.add(l1)
                locations.add(l2)
                self.location_graph.setdefault(l1, []).append(l2)
                self.location_graph.setdefault(l2, []).append(l1) # Links are bidirectional

        # Add locations from initial state and goal state that might not be linked
        # Ensure all locations mentioned in initial/goal states are nodes in the graph
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 locations.add(loc)
                 self.location_graph.setdefault(loc, []) # Ensure location exists in graph even if isolated

        for goal in self.goals:
             if match(goal, "at", "*", "*"): # Goals might specify location of objects
                 obj, loc = get_parts(goal)[1:]
                 locations.add(loc)
                 self.location_graph.setdefault(loc, [])

        # Ensure all locations found are in the graph keys, even if they have no links
        for loc in locations:
             self.location_graph.setdefault(loc, [])

        # Compute all-pairs shortest paths
        self.distances = {}
        all_locations = list(self.location_graph.keys())
        for start_loc in all_locations:
            self.distances[start_loc] = bfs(self.location_graph, start_loc)

        # Check if all goal nut locations are in the computed distances map
        # If not, they are unreachable from some locations, problem might be unsolvable.
        # We handle this by returning a large value in __call__.


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # Identify man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                man_location = get_parts(fact)[2]
                break

        if man_location is None:
             # Man's location not found in state - indicates an invalid state representation
             # print(f"Warning: Man location for '{self.man_name}' not found in state: {state}")
             return 1000000 # Large penalty

        # Identify loose goal nuts in the current state
        loose_goal_nuts_in_state = {
            nut for nut in self.goal_nuts
            if f"(loose {nut})" in state
        }

        # Identify usable spanners in the current state
        usable_spanners_in_state = {
            s for s in {get_parts(f)[1] for f in state if match(f, "usable", "*")}
        }

        # Identify spanners carried by the man in the current state
        carried_spanner = None
        for fact in state:
            if match(fact, "carrying", self.man_name, "*"):
                carried_spanner = get_parts(fact)[2]
                break

        # Identify usable spanners on the ground
        usable_spanners_on_ground = {
            s for s in usable_spanners_in_state
            if s != carried_spanner # Exclude the one being carried
        }

        # Get locations of usable spanners on the ground
        spanner_locations_on_ground = {}
        for spanner in usable_spanners_on_ground:
            for fact in state:
                if match(fact, "at", spanner, "*"):
                    spanner_locations_on_ground[spanner] = get_parts(fact)[2]
                    break # Found location for this spanner

        # Check if man is carrying a usable spanner
        is_carrying_usable_spanner = (carried_spanner is not None) and (carried_spanner in usable_spanners_in_state)

        h = 0
        # Process loose goal nuts in a fixed order (alphabetical)
        nuts_to_process = sorted(list(loose_goal_nuts_in_state))

        current_man_loc = man_location
        current_is_carrying_usable_spanner = is_carrying_usable_spanner
        # Create a mutable copy of available spanners on the ground for simulation
        simulated_available_spanners_on_ground = list(spanner_locations_on_ground.items()) # list of (spanner, location) tuples

        for nut in nuts_to_process:
            nut_loc = self.nut_locations.get(nut)
            if nut_loc is None:
                 # Location of goal nut not found (e.g., not in initial state 'at' facts)
                 # This indicates a problem with the problem definition or parsing.
                 # Return a large penalty.
                 # print(f"Warning: Location not found for goal nut: {nut}")
                 h += 1000000 # Large penalty
                 continue # Skip this nut

            # Check if nut location is reachable from man's current location
            if current_man_loc not in self.distances or nut_loc not in self.distances[current_man_loc] or self.distances[current_man_loc][nut_loc] == float('inf'):
                 # Nut location is unreachable from current man location
                 # This state is likely unsolvable.
                 # print(f"Warning: Nut location {nut_loc} unreachable from man location {current_man_loc}")
                 h += 1000000 * (len(nuts_to_process) - nuts_to_process.index(nut)) # Penalty for this and remaining
                 break # Cannot reach this nut

            # 1. Cost to move man to nut location
            if current_man_loc != nut_loc:
                h += self.distances[current_man_loc][nut_loc]
                current_man_loc = nut_loc

            # 2. Cost to get a usable spanner if needed
            if not current_is_carrying_usable_spanner:
                # Find the nearest available usable spanner on the ground from current_man_loc (which is nut_loc)
                min_dist_to_spanner = float('inf')
                best_spanner_info = None # (spanner, location)

                # Filter spanners to only include those whose location is reachable
                reachable_spanners = [(s, loc) for s, loc in simulated_available_spanners_on_ground
                                      if current_man_loc in self.distances and loc in self.distances[current_man_loc] and self.distances[current_man_loc][loc] != float('inf')]

                for spanner, spanner_loc in reachable_spanners:
                    dist = self.distances[current_man_loc][spanner_loc]
                    if dist < min_dist_to_spanner:
                        min_dist_to_spanner = dist
                        best_spanner_info = (spanner, spanner_loc)

                if best_spanner_info is not None:
                    spanner_to_pickup, spanner_loc = best_spanner_info

                    # Cost to move to spanner
                    h += min_dist_to_spanner
                    current_man_loc = spanner_loc # Man is now at spanner_loc

                    # Cost to pickup action
                    h += 1
                    current_is_carrying_usable_spanner = True
                    # Remove the used spanner from the available pool
                    simulated_available_spanners_on_ground.remove(best_spanner_info)

                    # Cost to move back to nut location (from spanner_loc)
                    if current_man_loc != nut_loc:
                         # Check reachability before adding distance
                         if current_man_loc in self.distances and nut_loc in self.distances[current_man_loc] and self.distances[current_man_loc][nut_loc] != float('inf'):
                             h += self.distances[current_man_loc][nut_loc]
                             current_man_loc = nut_loc
                         else:
                             # Should not happen if spanner location was reachable from nut_loc
                             # and nut_loc is reachable from spanner_loc (graph is undirected)
                             # Add penalty and break if somehow unreachable
                             # print(f"Warning: Nut location {nut_loc} unreachable from spanner location {current_man_loc} after pickup")
                             h += 1000000 * (len(nuts_to_process) - nuts_to_process.index(nut))
                             break # Cannot get back to nut

                else:
                    # No usable spanners left to pick up for this nut.
                    # This state is likely unsolvable from here.
                    # Add a large penalty for this nut and all remaining nuts.
                    # print(f"Warning: No usable spanners available for nut {nut}")
                    h += 1000000 * (len(nuts_to_process) - nuts_to_process.index(nut))
                    break # Cannot proceed with tightening

            # 3. Tighten the nut
            if current_is_carrying_usable_spanner: # Only if we successfully got/had a usable spanner
                h += 1 # Tighten action
                current_is_carrying_usable_spanner = False # Spanner becomes unusable

        return h
