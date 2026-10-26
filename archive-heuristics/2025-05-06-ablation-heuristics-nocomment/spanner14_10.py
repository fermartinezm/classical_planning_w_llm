import math
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper function to parse a fact string like '(predicate arg1 arg2)'
def parse_fact(fact_str):
    """Parses a PDDL fact string into a list of strings."""
    # Remove parentheses and split by spaces
    return fact_str[1:-1].split()

class spannerHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the Spanner domain.

    Summary:
        This heuristic estimates the cost to reach the goal by simulating a
        sequence of necessary actions for each loose goal nut. It assumes the
        agent needs to fetch a usable spanner and travel to a nut's location
        for each nut that needs tightening. The cost is the sum of the estimated
        costs for tightening each loose goal nut sequentially. For each nut,
        this involves:
        1. Walking to a usable spanner location (if not already carrying one).
        2. Picking up a spanner.
        3. Walking from the spanner location to the nut location.
        4. Tightening the nut.
        The heuristic calculates the cost for the first nut (including getting
        the first spanner if needed) and then adds the cost for each subsequent
        nut, assuming a new spanner must be fetched each time.

    Assumptions:
        - Nuts do not move from their initial locations.
        - Spanners become unusable after one use (`tighten_nut` action).
        - The agent can only effectively use one spanner at a time for tightening
          (even if the PDDL allows carrying multiple, the model assumes fetching
          a new one is needed for each subsequent nut).
        - The location graph defined by `link` predicates is static and represents
          all possible movement paths for the man.
        - The cost of each action (walk, pickup, tighten) is 1.

    Heuristic Initialization:
        In the constructor (`__init__`), the heuristic precomputes static information:
        - Identifies the man object name.
        - Identifies all location object names.
        - Identifies all nut object names.
        - Identifies all spanner object names.
        - Extracts the names of the goal nuts from the task definition.
        - Maps each nut name to its initial location based on the initial state.
        - Builds the location graph (adjacency list) from the `link` predicates in static facts.
        - Computes all-pairs shortest path distances between all locations using BFS.
          These distances represent the minimum number of `walk` actions required.

    Step-By-Step Thinking for Computing Heuristic:
        In the heuristic function (`__call__`), for a given state:
        1. Identify all loose nuts that are also goal nuts. If there are none, the heuristic is 0.
        2. Find the man's current location.
        3. Determine if the man is currently carrying a *usable* spanner.
        4. Identify all usable spanners currently at locations (not carried by man) and their locations.
        5. Check if there are enough usable spanners available (carried or at locations) to tighten all loose goal nuts. If not, the state is unsolvable, and the heuristic returns infinity.
        6. Initialize the heuristic value `h` to 0.
        7. Determine the man's effective starting location for the first phase.
           - If the man is *not* carrying a usable spanner: He needs to get one first. Find the usable spanner location closest to his current location (from those at locations). Add the distance to this spanner location plus 1 (for the pickup action) to `h`. Update the man's effective current location to the spanner location.
           - If the man *is* carrying a usable spanner: He is ready to go to the first nut. His effective current location remains his actual current location.
        8. Find the location of a remaining loose goal nut that is closest to the man's effective current location. Add the distance to this nut location to `h`. Update the man's effective current location to the nut location.
        9. Add 1 to `h` for the `tighten_nut` action for the first nut. Mark one nut at the current location as "tightened" for the purpose of counting remaining nuts.
        10. Enter a loop that continues as long as there are remaining loose goal nuts to tighten. For each remaining nut:
            - The man needs another usable spanner (as the previous one is assumed used and unusable). Re-identify usable spanners currently at locations from the current state. Find the usable spanner location closest to the man's current effective location. Add the distance to this spanner location plus 1 (for the pickup action) to `h`. Update the man's effective current location to the spanner location.
            - Find the location of a remaining loose goal nut that is closest to the man's current effective location. Add the distance to this nut location to `h`. Update the man's effective current location to the nut location.
            - Add 1 to `h` for the `tighten_nut` action. Mark one nut at the current location as "tightened" for the purpose of counting remaining nuts.
        11. Return the total heuristic value `h`.
    """
    def __init__(self, task):
        self.goals = task.goals
        self.static_facts = task.static
        self.objects = task.objects

        # Identify object types
        self.man_name = None
        self.location_names = []
        self.nut_names = []
        self.spanner_names = []

        for obj_group in self.objects:
            # obj_group is like ('bob', '-', 'man') or ('shed', 'location1', '-', 'location')
            # The format is (name1 name2 ... - type)
            parts = obj_group[0].split()
            type_name = parts[-1] # The last part is the type
            # Object names are the parts before '-'
            try:
                dash_index = parts.index('-')
                obj_names = parts[:dash_index]
            except ValueError:
                 # Handle cases like ('objname', 'type') if '-' is not present
                 # Based on example, '-' is always present before type
                 obj_names = [] # Should not happen based on examples

            if type_name == 'man':
                if obj_names: self.man_name = obj_names[0]
            elif type_name == 'location':
                self.location_names.extend(obj_names)
            elif type_name == 'nut':
                self.nut_names.extend(obj_names)
            elif type_name == 'spanner':
                self.spanner_names.extend(obj_names)

        # Identify goal nut names
        self.goal_nut_names = set()
        for goal in self.goals:
            parts = parse_fact(goal)
            if parts and parts[0] == 'tightened' and len(parts) > 1:
                self.goal_nut_names.add(parts[1])

        # Map nut names to their initial locations (nuts don't move)
        self.nut_initial_location = {}
        for fact in task.initial_state:
            parts = parse_fact(fact)
            if parts and parts[0] == 'at' and len(parts) > 2 and parts[1] in self.nut_names:
                 self.nut_initial_location[parts[1]] = parts[2]

        # Build location graph and compute distances
        self.distances = {loc: {l: float('inf') for l in self.location_names} for loc in self.location_names}
        for loc in self.location_names:
            self.distances[loc][loc] = 0

        adj_list = {loc: [] for loc in self.location_names}
        for fact in self.static_facts:
            parts = parse_fact(fact)
            if parts and parts[0] == 'link' and len(parts) > 2:
                l1, l2 = parts[1], parts[2]
                if l1 in adj_list and l2 in adj_list: # Ensure locations are valid
                    adj_list[l1].append(l2)
                    adj_list[l2].append(l1) # Links are bidirectional

        # Compute all-pairs shortest paths using BFS
        for start_node in self.location_names:
            q = deque([(start_node, 0)])
            visited = {start_node}
            while q:
                current_loc, dist = q.popleft()
                self.distances[start_node][current_loc] = dist

                for neighbor in adj_list.get(current_loc, []):
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append((neighbor, dist + 1))

    def get_distance(self, loc1, loc2):
        """Returns the precomputed shortest distance between two locations."""
        if loc1 is None or loc2 is None: return float('inf')
        return self.distances.get(loc1, {}).get(loc2, float('inf'))

    def get_man_location(self, state):
        """Finds the man's current location in the state."""
        for fact in state:
            parts = parse_fact(fact)
            if parts and parts[0] == 'at' and len(parts) > 1 and parts[1] == self.man_name:
                return parts[2]
        return None # Should not happen in valid states

    def is_man_carrying_usable_spanner(self, state):
        """Checks if the man is carrying any usable spanner."""
        carried_spanners = {parse_fact(fact)[2] for fact in state if parse_fact(fact) and parse_fact(fact)[0] == 'carrying' and len(parse_fact(fact)) > 2 and parse_fact(fact)[1] == self.man_name}
        usable_spanners_in_state = {parse_fact(fact)[1] for fact in state if parse_fact(fact) and parse_fact(fact)[0] == 'usable' and len(parse_fact(fact)) > 1}

        return any(s in usable_spanners_in_state for s in carried_spanners)


    def get_usable_spanner_locations(self, state):
        """Finds locations of usable spanners that are currently AT a location (not carried)."""
        usable_spanner_names_in_state = {parse_fact(fact)[1] for fact in state if parse_fact(fact) and parse_fact(fact)[0] == 'usable' and len(parse_fact(fact)) > 1 and parse_fact(fact)[1] in self.spanner_names}

        # Find spanners the man is carrying
        carried_spanner_names = {parse_fact(fact)[2] for fact in state if parse_fact(fact) and parse_fact(fact)[0] == 'carrying' and len(parse_fact(fact)) > 2 and parse_fact(fact)[1] == self.man_name}

        usable_spanners_at_loc = set()
        for spanner_name in usable_spanner_names_in_state:
            # Only consider usable spanners not currently carried by the man
            if spanner_name not in carried_spanner_names:
                 for fact in state:
                    parts = parse_fact(fact)
                    if parts and parts[0] == 'at' and len(parts) > 2 and parts[1] == spanner_name and parts[2] in self.location_names:
                        usable_spanners_at_loc.add(parts[2])
                        break # Spanner is only at one location
        return usable_spanners_at_loc

    def find_closest_location(self, from_loc, set_of_locations):
        """Finds the location in set_of_locations closest to from_loc."""
        min_dist = float('inf')
        closest_loc = None
        if not set_of_locations:
            return None, float('inf')

        for loc in set_of_locations:
            dist = self.get_distance(from_loc, loc)
            if dist < min_dist:
                min_dist = dist
                closest_loc = loc
        return closest_loc, min_dist

    def __call__(self, node):
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nut_names = {parse_fact(fact)[1] for fact in state if parse_fact(fact) and parse_fact(fact)[0] == 'loose' and len(parse_fact(fact)) > 1 and parse_fact(fact)[1] in self.goal_nut_names}
        nuts_remaining = list(loose_goal_nut_names) # Use a list to track remaining
        num_loose_goal_nuts = len(nuts_remaining)

        if num_loose_goal_nuts == 0:
            return 0

        # 2. Find man's current location
        man_loc = self.get_man_location(state)
        if man_loc is None: return float('inf') # Should not happen

        # 3. Determine if man is carrying a *usable* spanner
        carried_usable_spanner = self.is_man_carrying_usable_spanner(state)

        # 4. Identify usable spanners at locations
        usable_spanner_locations = self.get_usable_spanner_locations(state)

        # 5. Check if enough usable spanners exist in total (carried usable + at locations)
        num_usable_spanners_in_state = sum(1 for fact in state if parse_fact(fact) and parse_fact(fact)[0] == 'usable' and len(parse_fact(fact)) > 1 and parse_fact(fact)[1] in self.spanner_names)

        if num_usable_spanners_in_state < num_loose_goal_nuts:
             return float('inf') # Not enough spanners in the entire state

        h = 0
        current_loc = man_loc

        # Phase 1: Get the first spanner if needed, and go to the first nut
        if not carried_usable_spanner:
            # Find closest usable spanner location (from those at locations)
            closest_spanner_loc, dist_to_spanner = self.find_closest_location(current_loc, usable_spanner_locations)
            if closest_spanner_loc is None:
                 # This implies usable_spanner_locations was empty, but num_usable_spanners_in_state >= num_loose_goal_nuts > 0.
                 # This means all usable spanners are currently carried, but none are usable carried. This is a contradiction
                 # with the check `carried_usable_spanner` being False.
                 # Return inf as a safeguard.
                 return float('inf')

            h += dist_to_spanner + 1 # walk + pickup
            current_loc = closest_spanner_loc
            # Now effectively carrying a usable spanner for the next step

        # Find the location of a remaining loose goal nut closest to current_loc
        remaining_nut_locations = {self.nut_initial_location[nut] for nut in nuts_remaining if nut in self.nut_initial_location}
        closest_nut_loc, dist_to_nut = self.find_closest_location(current_loc, remaining_nut_locations)
        if closest_nut_loc is None:
             # Should not happen if nuts_remaining > 0 and nut_initial_location is correctly populated
             return float('inf')

        h += dist_to_nut # walk
        current_loc = closest_nut_loc

        # Tighten one nut at current_loc
        h += 1 # tighten
        # Remove one arbitrary nut from nuts_remaining that is at current_loc
        nut_tightened_here = None
        for nut in nuts_remaining:
            if self.nut_initial_location.get(nut) == current_loc:
                nut_tightened_here = nut
                break
        if nut_tightened_here:
            nuts_remaining.remove(nut_tightened_here)
        # After tightening, the spanner used is no longer usable.
        # For the heuristic model, assume the man needs to fetch another usable one for the next nut.

        # Phase 2: Get subsequent spanners and tighten remaining nuts
        while nuts_remaining:
            # Man needs another usable spanner.
            # Re-identify usable spanners at locations from the current state
            # Note: This heuristic doesn't perfectly track spanner consumption in the state representation
            # within the heuristic's simulation, but relies on the actual state facts.
            usable_spanner_locations_in_state = self.get_usable_spanner_locations(state)

            if not usable_spanner_locations_in_state:
                 # No more usable spanners at locations. If nuts remain, unsolvable.
                 return float('inf')

            # Find closest usable spanner location from current_loc
            closest_spanner_loc, dist_to_spanner = self.find_closest_location(current_loc, usable_spanner_locations_in_state)
            if closest_spanner_loc is None: return float('inf') # Should not happen

            h += dist_to_spanner + 1 # walk + pickup
            current_loc = closest_spanner_loc
            # Now effectively carrying a usable spanner for the next step

            # Find the location of a remaining loose goal nut closest to current_loc
            remaining_nut_locations = {self.nut_initial_location[nut] for nut in nuts_remaining if nut in self.nut_initial_location}
            closest_nut_loc, dist_to_nut = self.find_closest_location(current_loc, remaining_nut_locations)
            if closest_nut_loc is None: return float('inf') # Should not happen

            h += dist_to_nut # walk
            current_loc = closest_nut_loc

            # Tighten one nut at current_loc
            h += 1 # tighten
            # Remove one arbitrary nut from nuts_remaining that is at current_loc
            nut_tightened_here = None
            for nut in nuts_remaining:
                if self.nut_initial_location.get(nut) == current_loc:
                    nut_tightened_here = nut
                    break
            if nut_tightened_here:
                nuts_remaining.remove(nut_tightened_here)
            # After tightening, the spanner used is no longer usable.

        return h
