from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
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
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all loose nuts.
    It simulates a greedy strategy where the man always goes to the closest
    available usable spanner (if needed), picks it up, and then goes to the
    closest remaining loose nut and tightens it. The total cost is the sum
    of walk, pickup, and tighten actions in this simulated sequence.

    # Assumptions
    - Nuts are static; their location does not change. Their locations are determined from the initial state.
    - Spanners are static unless carried; their location on the ground does not change unless picked up. There is no action to drop a spanner.
    - A spanner is consumed (becomes not usable) after tightening one nut.
    - There is only one man.
    - The problem is solvable (enough usable spanners exist initially or can be obtained).

    # Heuristic Initialization
    - Store the goal conditions.
    - Build a graph of locations based on `link` facts from static information.
    - Compute all-pairs shortest path distances between locations using BFS on the location graph.
    - Identify all nuts and their locations from the initial state (assuming nuts are static).

    # Step-By-Step Thinking for Computing Heuristic
    1. Check if the current state is a goal state. If yes, the heuristic is 0.
    2. Extract the current location of the man from the state.
    3. Identify which usable spanners are currently carried by the man from the state.
    4. Identify which usable spanners are currently on the ground and their locations from the state.
    5. Identify which nuts are currently loose from the state.
    6. Initialize the total estimated cost to 0.
    7. Initialize the man's current location for the simulation to his actual current location.
    8. Initialize the number of spanners carried by the man for the simulation based on the current state.
    9. Initialize the dictionary of available usable spanners on the ground and their locations for the simulation based on the current state.
    10. Initialize the set of remaining loose nuts for the simulation based on the current state.
    11. While there are still loose nuts remaining in the simulation:
        a. If the man is not currently carrying a usable spanner in the simulation:
           i. Check if there are any usable spanners available on the ground in the simulation. If not, the state is likely unsolvable by this heuristic's logic; return a large cost.
           ii. Find the usable spanner on the ground that is closest to the man's current simulation location.
           iii. Add the walk distance to this spanner's location to the total cost.
           iv. Add 1 to the total cost for the pickup action.
           v. Update the man's current simulation location to the spanner's location.
           vi. Mark the spanner as carried (remove it from the available ground spanners and increment the carried count).
        b. Now the man is carrying a spanner in the simulation. Find the remaining loose nut that is closest to the man's current simulation location.
        c. Add the walk distance to this nut's location (which is static) to the total cost.
        d. Add 1 to the total cost for the tighten action.
        e. Update the man's current simulation location to the nut's location.
        f. Mark the spanner as used (decrement the carried count).
        g. Mark the nut as tightened (remove it from the set of remaining loose nuts).
    12. Return the total estimated cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and precomputing distances.
        """
        self.goals = task.goals
        self.static = task.static

        # Build location graph and compute distances
        self.adj = {}
        locations = set()
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)[1:]
                locations.add(loc1)
                locations.add(loc2)
                self.adj.setdefault(loc1, []).append(loc2)
                self.adj.setdefault(loc2, []).append(loc1) # Links are bidirectional

        self.locations = list(locations)
        self.distance = {}

        # Compute all-pairs shortest paths using BFS
        for start_loc in self.locations:
            self.distance[start_loc] = {}
            q = deque([(start_loc, 0)])
            visited = {start_loc}
            while q:
                current_loc, dist = q.popleft()
                self.distance[start_loc][current_loc] = dist
                for neighbor in self.adj.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

        # Identify all nuts and their initial locations (assumed static)
        self.nut_locations = {}
        # Access initial state from the task object
        initial_state = task.initial_state
        for fact in initial_state:
            if match(fact, "at", "?n", "?l"):
                obj, loc = get_parts(fact)[1:]
                # Assume objects starting with 'nut' are nuts based on domain conventions
                if obj.startswith('nut'):
                    self.nut_locations[obj] = loc

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions
        using a greedy simulation.
        """
        state = node.state

        # Check if goal is reached
        if self.goals <= state:
            return 0

        # --- Extract current state information ---
        man_loc = None
        carried_spanners = set()
        usable_spanners_in_state = set() # Usable spanners regardless of location
        loose_nuts = set()

        # Map current locations of usable spanners on the ground
        current_usable_ground_spanner_locations = {}

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                if obj.startswith('man'): # Assuming only one man
                    man_loc = loc
                # Nut locations are assumed static, handled in __init__

            elif parts[0] == 'carrying':
                 # Assuming the second argument is the spanner
                 carried_spanners.add(parts[2])
            elif parts[0] == 'usable':
                 usable_spanners_in_state.add(parts[1])
            elif parts[0] == 'loose':
                 loose_nuts.add(parts[1])

        # Identify usable spanners currently carried
        usable_carried_spanners = carried_spanners.intersection(usable_spanners_in_state)

        # Identify usable spanners currently on the ground and their locations
        for fact in state:
             if match(fact, "at", "?s", "?l"):
                 obj, loc = get_parts(fact)[1:]
                 # Assume objects starting with 'spanner' are spanners
                 if obj.startswith('spanner') and obj in usable_spanners_in_state:
                     current_usable_ground_spanner_locations[obj] = loc


        # --- Greedy Simulation ---
        current_sim_loc = man_loc
        sim_num_spanners_carried = len(usable_carried_spanners)
        # Use a dictionary copy for simulation, mapping spanner name to its current ground location
        sim_available_ground_spanners_locs = dict(current_usable_ground_spanner_locations)
        sim_remaining_loose_nuts = set(loose_nuts)

        cost = 0

        while sim_remaining_loose_nuts:
            # Need to tighten one nut
            if sim_num_spanners_carried == 0:
                # Need to pick up a spanner
                if not sim_available_ground_spanners_locs:
                    # This state is likely unsolvable within the heuristic's logic
                    # Return a large value to penalize it heavily
                    return 1000000 # Indicate difficulty/unsolvability

                # Find closest usable spanner on ground
                closest_spanner_name = None
                min_dist = float('inf')
                for s_name, s_loc in sim_available_ground_spanners_locs.items():
                    # Ensure the location is in our distance map (graph might be disconnected, though unlikely in typical problems)
                    if current_sim_loc not in self.distance or s_loc not in self.distance[current_sim_loc]:
                         # Cannot reach this spanner location
                         continue

                    dist = self.distance[current_sim_loc][s_loc]
                    if dist < min_dist:
                        min_dist = dist
                        closest_spanner_name = s_name

                if closest_spanner_name is None:
                     # This happens if sim_available_ground_spanners_locs is not empty,
                     # but none of the spanner locations are reachable from current_sim_loc.
                     # Also indicates unsolvability or a very difficult state.
                     return 1000001 # Error case / Unsolvable

                # Walk to spanner and pick it up
                cost += min_dist # Walk action cost
                cost += 1 # Pickup action cost
                current_sim_loc = sim_available_ground_spanners_locs[closest_spanner_name]
                del sim_available_ground_spanners_locs[closest_spanner_name] # Remove from available ground spanners
                sim_num_spanners_carried += 1

            # Now we have a spanner (sim_num_spanners_carried > 0)
            # Find the closest remaining loose nut
            closest_nut_name = None
            min_dist = float('inf')
            for n_name in sim_remaining_loose_nuts:
                nut_loc = self.nut_locations.get(n_name) # Use static nut location
                if nut_loc is None:
                    # Error case / Nut location not found (should be in self.nut_locations from initial state)
                    return 1000002

                # Ensure the nut location is in our distance map
                if current_sim_loc not in self.distance or nut_loc not in self.distance[current_sim_loc]:
                     # Cannot reach this nut location
                     continue

                dist = self.distance[current_sim_loc][nut_loc]
                if dist < min_dist:
                    min_dist = dist
                    closest_nut_name = n_name

            if closest_nut_name is None:
                 # This happens if sim_remaining_loose_nuts is not empty,
                 # but none of the nut locations are reachable from current_sim_loc.
                 # Also indicates unsolvability or a very difficult state.
                 return 1000003 # Error case

            # Walk to nut and tighten it
            cost += min_dist # Walk action cost
            cost += 1 # Tighten action cost
            current_sim_loc = self.nut_locations[closest_nut_name]
            sim_num_spanners_carried -= 1 # Spanner used
            sim_remaining_loose_nuts.remove(closest_nut_name)

        return cost
