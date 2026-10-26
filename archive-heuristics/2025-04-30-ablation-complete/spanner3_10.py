from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty fact strings or malformed facts defensively
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
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
    This heuristic estimates the number of actions required to tighten all
    goal nuts. It considers the cost of walking to locations, picking up
    usable spanners, and performing the tightening action. It uses a greedy
    approach to estimate movement costs, prioritizing getting a spanner
    when needed and then going to the closest available loose nut.

    # Assumptions
    - There is exactly one man object.
    - Nuts stay in their initial locations.
    - Spanners become unusable after one tightening action.
    - A man can only carry one spanner at a time (implicit in action structure).
    - Links between locations are bidirectional.
    - The graph of locations is connected for solvable problems.
    - The man object name can be inferred from the initial state.

    # Heuristic Initialization
    - Builds a graph of locations based on `link` predicates.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Identifies the name of the man object from the initial state.
    - Stores goal nuts.

    # Step-By-Step Thinking for Computing Heuristic
    For a given state:
    1. Identify the set of loose nuts that need to be tightened (i.e., are loose in the current state and are goal conditions). If this set is empty, the heuristic is 0.
    2. Find the man's current location.
    3. Determine if the man is currently carrying a usable spanner.
    4. Identify the locations of all usable spanners that are currently on the ground (not carried).
    5. Initialize the heuristic cost `h` to 0.
    6. Initialize the man's current location for the simulation (`current_loc_sim`).
    7. Initialize the state of carrying a usable spanner for the simulation (`has_usable_spanner_sim`).
    8. Initialize the list of nuts that still need to be tightened in the simulation (`nuts_to_do_sim`).
    9. Initialize the list of locations where usable spanners are available on the ground for pickup in the simulation (`available_spanner_locations_sim`).
    10. While there are still nuts to tighten in the simulation (`nuts_to_do_sim` is not empty):
        a. If the man is not currently simulated as carrying a usable spanner (`has_usable_spanner_sim` is False):
            i. Find the closest location with a usable spanner from the `available_spanner_locations_sim`. If none exist, the problem is likely unsolvable from this state, return infinity.
            ii. Add the walk distance to this spanner location to `h`.
            iii. Add 1 to `h` for the `pickup_spanner` action.
            iv. Update `current_loc_sim` to the spanner location.
            v. Set `has_usable_spanner_sim` to True.
            vi. Remove the used spanner location from `available_spanner_locations_sim` for the simulation.
        b. If the man is currently simulated as carrying a usable spanner (`has_usable_spanner_sim` is True):
            i. Find the closest location of a nut from `nuts_to_do_sim`. If none exist (shouldn't happen if the outer loop condition is correct), return infinity.
            ii. Add the walk distance to this nut location to `h`.
            iii. Add 1 to `h` for the `tighten_nut` action.
            iv. Update `current_loc_sim` to the nut location.
            v. Remove the tightened nut from `nuts_to_do_sim`.
            vi. Set `has_usable_spanner_sim` to False (spanner used).
    11. Return the total accumulated cost `h`.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static information and computing
        distances.
        """
        super().__init__(task)
        self.goals = task.goals
        self.initial_state = task.initial_state
        self.static_facts = task.static

        # Build location graph and compute distances
        self.locations = set()
        self.links = {} # Adjacency list: location -> set of connected locations
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.locations.add(l1)
                self.locations.add(l2)
                self.links.setdefault(l1, set()).add(l2)
                self.links.setdefault(l2, set()).add(l1) # Links are bidirectional

        self.distances = {} # (loc1, loc2) -> distance
        for start_loc in self.locations:
            q = [(start_loc, 0)]
            visited = {start_loc}
            self.distances[(start_loc, start_loc)] = 0
            head = 0
            while head < len(q):
                current_loc, dist = q[head]
                head += 1
                if current_loc in self.links:
                    for neighbor in self.links[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_loc, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

        # Find the man object name from initial state facts
        self.man_name = None
        # Attempt to find the man object name. This is fragile and depends on
        # the structure of the initial state and the PDDL problem definition
        # not being fully available in the Task object.
        # A robust solution requires a proper PDDL parser providing object types.

        initial_nuts = {get_parts(f)[1] for f in self.initial_state if get_parts(f) and get_parts(f)[0] == 'loose'}
        goal_nuts = {get_parts(g)[1] for g in self.goals if get_parts(g) and get_parts(g)[0] == 'tightened'}
        all_nuts = initial_nuts.union(goal_nuts)

        initial_spanners = set()
        for fact in self.initial_state:
             parts = get_parts(fact)
             if parts:
                 if parts[0] == 'usable':
                     initial_spanners.add(parts[1])
                 elif parts[0] == 'at' and 'spanner' in parts[1].lower(): # Heuristic guess
                     initial_spanners.add(parts[1])
                 elif parts[0] == 'carrying':
                     # The carried object is likely a spanner in this domain
                     initial_spanners.add(parts[2])


        locatable_objects_in_init = {get_parts(f)[1] for f in self.initial_state if get_parts(f) and get_parts(f)[0] == 'at'}

        potential_men = locatable_objects_in_init - all_nuts - initial_spanners

        if len(potential_men) == 1:
             self.man_name = list(potential_men)[0]
        else:
             # Fallback: look for 'carrying' predicate in initial state
             for fact in self.initial_state:
                 parts = get_parts(fact)
                 if parts and parts[0] == 'carrying':
                     self.man_name = parts[1]
                     break

        # If man_name is still None, the heuristic cannot function correctly.
        # This indicates an issue with inferring the man object from the task representation.
        # The heuristic will likely return infinity or error later if man_location is not found.


    def get_distance(self, loc1, loc2):
        """Returns the shortest path distance between two locations."""
        # Handle cases where locations might not be in the graph or graph is disconnected.
        # For this domain, all locatables are expected to be at linked locations,
        # and the graph should be connected in solvable instances.
        # If not connected, distance is infinite.
        return self.distances.get((loc1, loc2), float('inf'))

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify nuts to tighten
        goal_nuts = {get_parts(g)[1] for g in self.goals if get_parts(g) and get_parts(g)[0] == 'tightened'}
        loose_nuts_in_state = {get_parts(f)[1] for f in state if get_parts(f) and get_parts(f)[0] == 'loose'}
        nuts_to_tighten_initial = list(goal_nuts.intersection(loose_nuts_in_state)) # Use a list for simulation

        if not nuts_to_tighten_initial:
            return 0 # All required nuts are tightened

        # Find object locations in the current state
        object_locations = {} # Map object -> location
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'at':
                obj, loc = parts[1], parts[2]
                object_locations[obj] = loc

        # Find man's current location
        man_location = object_locations.get(self.man_name)
        if man_location is None:
             # Man is not 'at' any location? Should not happen in this domain.
             # Or man_name was not correctly identified.
             return float('inf') # Indicate unsolvable or invalid state

        # Determine if man is carrying a usable spanner
        carrying_usable_spanner = False
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'carrying' and parts[1] == self.man_name:
                carried_spanner = parts[2]
                if '(usable ' + carried_spanner + ')' in state:
                    carrying_usable_spanner = True
                break # Assuming one man, one carried item

        # Identify locations of usable spanners on the ground in the current state
        usable_spanners_on_ground_locs = set()
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'usable':
                spanner_name = parts[1]
                # Check if this spanner is on the ground (not carried)
                is_carried = False
                for f in state:
                     p = get_parts(f)
                     if p and p[0] == 'carrying' and p[2] == spanner_name:
                         is_carried = True
                         break
                if not is_carried and spanner_name in object_locations:
                     usable_spanners_on_ground_locs.add(object_locations[spanner_name])

        # --- Greedy Simulation to Estimate Cost ---
        h = 0
        current_loc_sim = man_location
        has_usable_spanner_sim = carrying_usable_spanner
        nuts_to_do_sim = list(nuts_to_tighten_initial) # Copy the list
        available_spanner_locations_sim = list(usable_spanners_on_ground_locs) # Copy the list of locations

        while nuts_to_do_sim:
            if not has_usable_spanner_sim:
                # Need to pick up a spanner
                if not available_spanner_locations_sim:
                    # No spanners left to pick up for remaining nuts
                    return float('inf')

                # Find closest spanner location from the available locations in simulation
                min_dist_s = float('inf')
                closest_s_loc = None
                for s_loc in available_spanner_locations_sim:
                    dist = self.get_distance(current_loc_sim, s_loc)
                    if dist != float('inf') and dist < min_dist_s:
                        min_dist_s = dist
                        closest_s_loc = s_loc

                if closest_s_loc is None:
                     # Cannot reach any available spanner location
                     return float('inf')

                h += min_dist_s # Walk to spanner
                h += 1 # Pickup action
                current_loc_sim = closest_s_loc
                has_usable_spanner_sim = True
                # Remove the used spanner location from the available list in simulation
                available_spanner_locations_sim.remove(closest_s_loc)


            # Now carrying a spanner, go to the closest nut
            min_dist_n = float('inf')
            next_nut = None
            next_nut_loc = None
            for nut_name in nuts_to_do_sim:
                # Get the current location of the nut from the *actual* state (nuts don't move)
                nut_loc = object_locations.get(nut_name)
                if nut_loc is None:
                    # Nut location not found in state? Problematic state representation.
                    return float('inf') # Or handle error

                dist = self.get_distance(current_loc_sim, nut_loc)
                if dist != float('inf') and dist < min_dist_n:
                    min_dist_n = dist
                    next_nut = nut_name
                    next_nut_loc = nut_loc

            if next_nut is None:
                 # Should not happen if nuts_to_do_sim is not empty and graph is connected
                 return float('inf')

            h += min_dist_n # Walk to nut
            h += 1 # Tighten action
            current_loc_sim = next_nut_loc
            nuts_to_do_sim.remove(next_nut)
            has_usable_spanner_sim = False # Spanner used, need another for the next nut

        return h
