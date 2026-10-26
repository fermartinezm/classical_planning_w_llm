from fnmatch import fnmatch
# from heuristics.heuristic_base import Heuristic # Assuming this is available

# Define a dummy Heuristic base class if not running in the planner environment
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            self.goals = task.goals
            self.static = task.static
            self.objects = task.objects # Assume task.objects is available {obj_name: obj_type}

        def __call__(self, node):
            raise NotImplementedError


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    - `fact`: The complete fact as a string, e.g., "(in-city airport1 city1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions needed to tighten all goal nuts.
    It considers the cost of tightening actions, spanner pickup actions, and the
    travel cost for the man to visit nut locations and spanner locations.
    The travel cost is estimated by summing distances along a greedy path that
    alternates between visiting the closest remaining nut and picking up the
    closest available usable spanner when needed.

    # Assumptions:
    - There is exactly one man.
    - Links between locations are bidirectional.
    - A man can carry at most one spanner at a time (pickup replaces carried, implicitly).
    - Using a spanner makes it unusable.
    - All goal nuts are initially loose.
    - Solvable problems have enough usable spanners available initially.

    # Heuristic Initialization
    - Identify the man object.
    - Identify all locations and precompute all-pairs shortest path distances
      between them based on the 'link' predicates using BFS.
    - Store the goal conditions to identify the nuts that need tightening.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify the man's current location.
    2. Identify all loose nuts that are part of the goal and their current locations.
    3. If there are no loose goal nuts, the state is a goal state, return 0.
    4. Identify all usable spanners (carried or on the ground) and their locations.
    5. Check if the total number of usable spanners available is less than the number
       of loose goal nuts. If so, the problem is likely unsolvable, return a large value.
    6. Initialize the heuristic value:
       - Add the number of loose goal nuts (for the 'tighten_nut' actions).
       - Calculate the number of spanner pickups needed from the ground: This is equal
         to the number of loose goal nuts, unless the man is currently carrying a
         usable spanner, in which case it's one less (minimum 0). Add this number
         to the heuristic value (for the 'pickup_spanner' actions).
    7. Calculate the travel cost:
       - Initialize a sequence of locations to visit, starting empty.
       - Initialize current location for greedy search to the man's location.
       - If the man is *not* carrying a usable spanner (and needs at least one spanner):
         - Find the closest usable spanner on the ground from the current search location.
         - Add this spanner's location to the sequence.
         - Update the current search location to the spanner's location.
         - Remove this spanner from the list of available usable spanners on the ground.
         - If any required location is unreachable, return a large value.
       - Loop for each loose goal nut (number of times equals the total number of loose goal nuts):
         - Find the closest remaining loose goal nut from the current search location.
         - Add this nut's location to the sequence.
         - Update the current search location to the nut's location.
         - Remove this nut from the list of remaining loose nuts.
         - If any required location is unreachable, return a large value.
         - If we still need to pick up more spanners from the ground after processing this nut:
           - Find the closest available usable spanner on the ground from the current search location.
           - Add this spanner's location to the sequence.
           - Update the current search location to the spanner's location.
           - Remove this spanner from the list of available usable spanners on the ground.
           - If any required location is unreachable, return a large value.
       - Calculate the total travel cost by summing the shortest path distances between consecutive locations in the sequence, starting from the man's initial location.
    8. Add the total calculated travel cost to the heuristic value.
    9. Return the total heuristic value.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions, static facts, and precomputing distances."""
        super().__init__(task) # Call base class constructor
        static_facts = task.static
        self.objects = task.objects # Store objects to get types

        # Identify the man object (assuming exactly one man)
        self.man_objects = {obj for obj, obj_type in self.objects.items() if obj_type == 'man'}
        assert len(self.man_objects) == 1, "Heuristic assumes exactly one man."
        self.the_man = list(self.man_objects)[0]

        # Identify goal nuts
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'tightened'}

        # Build location graph and precompute all-pairs shortest paths
        self.location_links = {}
        all_locations = set()
        for fact in static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                l1, l2 = parts[1], parts[2]
                self.location_links.setdefault(l1, []).append(l2)
                self.location_links.setdefault(l2, []).append(l1) # Links are bidirectional
                all_locations.add(l1)
                all_locations.add(l2)
        self.all_locations = list(all_locations)

        self.all_pairs_distances = {}
        for start_loc in self.all_locations:
            self.all_pairs_distances[start_loc] = self.bfs_distances(start_loc, self.location_links, self.all_locations)

    def bfs_distances(self, start_location, links, locations):
        """Compute shortest path distances from a start location to all other locations using BFS."""
        distances = {loc: float('inf') for loc in locations}
        if start_location not in locations:
             # If the start location is not in the graph (e.g., man starts at an isolated location)
             # all distances from it will remain infinity. This is correct.
             pass
        else:
            distances[start_location] = 0
            queue = [start_location]
            while queue:
                current = queue.pop(0)
                if current in links:
                    for neighbor in links[current]:
                        if distances[neighbor] == float('inf'):
                            distances[neighbor] = distances[current] + 1
                            queue.append(neighbor)
        return distances


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state
        LARGE_VALUE = 1000000 # Value indicating unsolvability or very high cost

        # 1. Identify man's current location
        man_location = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.the_man:
                man_location = parts[2]
                break
        if man_location is None: return LARGE_VALUE # Invalid state

        # 2. Identify loose goal nuts and their locations
        loose_goal_nuts = {} # {nut_name: location}
        current_nut_locations = {} # {nut_name: location}
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at' and parts[1] in self.objects and self.objects[parts[1]] == 'nut':
                  current_nut_locations[parts[1]] = parts[2]

        for nut_name in self.goal_nuts:
             if f'(loose {nut_name})' in state:
                  if nut_name in current_nut_locations:
                       loose_goal_nuts[nut_name] = current_nut_locations[nut_name]
                  else: return LARGE_VALUE # Loose goal nut location not found

        # 3. If there are no loose goal nuts, it's a goal state
        if not loose_goal_nuts:
            return 0

        # 4. Identify usable spanners and their locations/carried status
        available_spanners_on_ground = [] # List of (spanner_name, location)
        carried_spanner = None

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'carrying' and parts[2] in self.objects and self.objects[parts[2]] == 'spanner':
                 carried_spanner = parts[2] # Assuming one carried spanner

        man_carrying_usable = False
        if carried_spanner is not None and f'(usable {carried_spanner})' in state:
             man_carrying_usable = True

        spanner_locations = {} # {spanner_name: location}
        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'at' and parts[1] in self.objects and self.objects[parts[1]] == 'spanner':
                  spanner_locations[parts[1]] = parts[2]

        for fact in state:
             parts = get_parts(fact)
             if parts[0] == 'usable' and parts[1] in spanner_locations:
                  available_spanners_on_ground.append((parts[1], spanner_locations[parts[1]]))

        # 5. Check if enough usable spanners exist in total
        num_loose_nuts = len(loose_goal_nuts)
        num_usable_available_total = len(available_spanners_on_ground) + (1 if man_carrying_usable else 0)

        if num_usable_available_total < num_loose_nuts:
             return LARGE_VALUE # Unsolvable

        # 6. Initialize the heuristic value
        h = num_loose_nuts # Cost for 'tighten_nut' actions

        # Calculate the number of spanner pickups needed from the ground
        num_spanners_to_pickup_from_ground = num_loose_nuts - (1 if man_carrying_usable else 0)
        h += max(0, num_spanners_to_pickup_from_ground) # Cost for 'pickup_spanner' actions

        # 7. Calculate the travel cost along the greedy path
        travel_cost = 0
        # Initialize current location for greedy search
        current_search_loc = man_location
        remaining_nut_names = list(loose_goal_nuts.keys())
        available_spanners_on_ground_mutable = list(available_spanners_on_ground)

        # Build the sequence of locations to visit greedily
        locations_sequence = []

        # If not carrying usable, the first stop is a spanner (if any pickup is needed)
        if not man_carrying_usable and num_spanners_to_pickup_from_ground > 0:
             # Find closest available spanner on ground from current_search_loc
             closest_s_info = None
             min_dist = float('inf')
             for s_info in available_spanners_on_ground_mutable:
                 dist = self.all_pairs_distances.get(current_search_loc, {}).get(s_info[1], float('inf'))
                 if dist == float('inf'): return LARGE_VALUE # Unreachable spanner
                 if dist < min_dist:
                     min_dist = dist
                     closest_s_info = s_info

             if closest_s_info is None: return LARGE_VALUE # Should have enough spanners

             locations_sequence.append(closest_s_info[1])
             current_search_loc = closest_s_info[1]
             available_spanners_on_ground_mutable.remove(closest_s_info)

        # Alternate between nuts and spanners
        for i in range(num_loose_nuts):
            # Go to the next nut (Ni)
            closest_nut_name = None
            min_dist = float('inf')
            # Find closest remaining nut from the current_search_loc
            for nut_name in remaining_nut_names:
                nut_loc = loose_goal_nuts[nut_name]
                dist = self.all_pairs_distances.get(current_search_loc, {}).get(nut_loc, float('inf'))
                if dist == float('inf'): return LARGE_VALUE # Unreachable nut
                if dist < min_dist:
                    min_dist = dist
                    closest_nut_name = nut_name

            if closest_nut_name is None: return LARGE_VALUE # Should not happen

            nut_loc = loose_goal_nuts[closest_nut_name]
            locations_sequence.append(nut_loc)
            current_search_loc = nut_loc
            remaining_nut_names.remove(closest_nut_name)

            # If we still need to pick up more spanners from the ground after processing this nut
            # Number of spanners picked up from ground so far = (1 if not carrying usable else 0) + i
            # We need another pickup if this count is less than the total needed from ground.
            num_ground_pickups_done_before_this_step = (1 if not man_carrying_usable else 0) + i
            if num_ground_pickups_done_before_this_step < num_spanners_to_pickup_from_ground:
                 # Find closest available spanner on ground from current_search_loc
                 closest_s_info = None
                 min_dist = float('inf')
                 for s_info in available_spanners_on_ground_mutable:
                     dist = self.all_pairs_distances.get(current_search_loc, {}).get(s_info[1], float('inf'))
                     if dist == float('inf'): return LARGE_VALUE # Unreachable spanner
                     if dist < min_dist:
                         min_dist = dist
                         closest_s_info = s_info

                 if closest_s_info is None:
                      # Should not happen if enough spanners exist and are reachable
                      return LARGE_VALUE # Error state or unsolvable

                 locations_sequence.append(closest_s_info[1])
                 current_search_loc = closest_s_info[1]
                 available_spanners_on_ground_mutable.remove(closest_s_info)


        # Calculate total travel cost by summing distances along the sequence
        current_loc_for_travel = man_location
        for next_loc in locations_sequence:
            dist = self.all_pairs_distances.get(current_loc_for_travel, {}).get(next_loc, float('inf'))
            if dist == float('inf'): return LARGE_VALUE # Should have been caught earlier, but safety check
            travel_cost += dist
            current_loc_for_travel = next_loc


        # 8. Add the total calculated travel cost to the heuristic value.
        h += travel_cost

        # 9. Return the total heuristic value.
        return h
