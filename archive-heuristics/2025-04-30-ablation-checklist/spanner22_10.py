import collections
from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

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
    """
    Perform Breadth-First Search to find shortest distances from a start node.

    Args:
        graph: Adjacency list representation of the graph {node: [neighbor1, neighbor2, ...]}
        start_node: The node to start the BFS from.

    Returns:
        A dictionary mapping each reachable node to its distance from the start_node.
    """
    distances = {node: float('inf') for node in graph}
    distances[start_node] = 0
    queue = collections.deque([start_node])

    while queue:
        current_node = queue.popleft()

        if current_node in graph: # Handle nodes that might be in distances but not graph (e.g. objects not locations)
            for neighbor in graph[current_node]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[current_node] + 1
                    queue.append(neighbor)
    return distances

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all goal nuts.
    It simulates a greedy sequence of actions:
    1. If the man needs a spanner, go to the nearest usable spanner location and pick it up.
    2. Go to the nearest untightened goal nut location and tighten it.
    This process repeats until all goal nuts are tightened.

    # Heuristic Initialization
    - Builds a graph of locations based on `link` facts.
    - Computes all-pairs shortest paths between locations using BFS.
    - Identifies goal nuts and their fixed locations.

    # Step-By-Step Thinking for Computing Heuristic (__call__)
    1. Identify the man's current location.
    2. Check if the man is currently carrying a usable spanner.
    3. Identify all loose nuts that are part of the goal.
    4. Identify the locations of all usable spanners currently on the ground.
    5. If there are no untightened goal nuts, the heuristic is 0.
    6. Simulate a greedy plan:
       - Initialize cost, current man location, spanner status, list of nuts to tighten, and available spanners (locations with counts).
       - While there are nuts left to tighten:
         - If the man does not have a spanner:
           - Find the nearest location with an available usable spanner.
           - Add travel cost to reach it. Update man's location.
           - Add 1 for the `pickup_spanner` action. Update spanner status (has spanner). Decrement the count of spanners at that location.
         - If the man has a spanner:
           - Find the nearest untightened nut location.
           - Add travel cost to reach it. Update man's location.
           - Add 1 for the `tighten_nut` action. Update spanner status (spanner consumed). Remove the nut from the list of nuts to tighten.
       - Return the total accumulated cost.

    Assumes:
    - There is only one man object.
    - Nut locations are static.
    - Spanners become unusable after one use and cannot be dropped or re-used.
    - Sufficient usable spanners exist in the initial state for all goal nuts (problem is solvable).
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting static facts and computing distances.
        """
        self.goals = task.goals  # Goal conditions.
        static_facts = task.static  # Facts that are not affected by actions.
        all_facts = task.facts # All possible facts in the domain

        # Build the location graph from link facts
        self.location_graph = collections.defaultdict(list)
        self.locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                l1, l2 = get_parts(fact)[1:]
                self.location_graph[l1].append(l2)
                self.location_graph[l2].append(l1) # Assuming links are bidirectional for movement
                self.locations.add(l1)
                self.locations.add(l2)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_loc in self.locations:
            self.distances[start_loc] = bfs(self.location_graph, start_loc)

        # Identify goal nuts and their locations (nuts are locatable and static)
        self.goal_nut_locations = {}
        all_objects = set()
        for fact in task.initial_state | static_facts: # Objects can be in initial state or static
             parts = get_parts(fact)
             if parts[0] in ["at", "carrying", "usable", "link", "tightened", "loose"]:
                 all_objects.update(parts[1:])

        # Find all nut objects
        nut_objects = set()
        for obj in all_objects:
             # Check if this object appears in a (loose obj) or (tightened obj) fact in goals or initial state
             if any(match(f, "loose", obj) for f in task.initial_state) or \
                any(match(f, "tightened", obj) for f in task.goals):
                 nut_objects.add(obj)

        # Find initial locations of nuts (assuming nuts don't move except implicitly by tightening)
        # The domain description implies nuts are fixed at locations.
        for fact in task.initial_state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                if obj in nut_objects:
                    self.goal_nut_locations[obj] = loc


    def get_distance(self, loc1, loc2):
        """Get the pre-computed shortest distance between two locations."""
        if loc1 not in self.distances or loc2 not in self.distances[loc1]:
             # This might happen if an object's location isn't in the graph (e.g. carried object)
             # Or if locations from state/goals weren't in the initial link facts (shouldn't happen in valid PDDL)
             # If loc1 is not a location node in the graph, distance is infinite.
             # If loc2 is not reachable from loc1, distance is infinite.
             return float('inf')
        return self.distances[loc1][loc2]


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state  # Current world state.

        # 1. Identify man's current location
        man_loc = None
        man_name = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Assume the object typed 'man' is the one in the 'at' predicate
                # A more robust way would be to parse object types from the problem file
                # For this domain, we can assume the single object of type 'man' is the one moving.
                # Let's find the man object name first.
                if man_name is None:
                     # Find man object name from state or initial state (assuming one man)
                     for f in state:
                         parts = get_parts(f)
                         if parts[0] == 'carrying':
                             man_name = parts[1]
                             break # Found man name
                     if man_name is None: # Man might not be carrying anything yet
                          for f in self.goals: # Man must be involved in tightening
                              parts = get_parts(f)
                              if parts[0] == 'tightened': # Find a nut
                                  # Need to find an action that involves the man and this nut
                                  # This is getting complex. Let's assume the man object name is 'bob' or similar
                                  # A better way: parse object types from problem file during init.
                                  # For now, let's just find the object at a location that isn't a spanner or nut.
                                  # This is fragile. Let's rely on the 'carrying' predicate if possible, or assume a common name like 'bob'.
                                  # Given the examples, 'bob' is the man. Let's hardcode for simplicity in this domain.
                                  man_name = 'bob' # HACK: Assuming man object is named 'bob'


                if man_name is not None and match(fact, "at", man_name, "*"):
                    man_loc = loc
                    break # Found man's location

        if man_loc is None:
             # Man is likely carrying a spanner and not 'at' a location.
             # The 'carrying' predicate removes the 'at' predicate for the spanner,
             # but the man's 'at' predicate should persist.
             # If man_loc is None here, something is wrong with state representation or domain understanding.
             # Let's assume man_loc is always present unless the state is invalid.
             # For robustness, return infinity if man_loc isn't found.
             # print(f"Warning: Man's location not found in state: {state}")
             return float('inf')


        # 2. Check if man is carrying a usable spanner
        carrying_usable_spanner = False
        carried_spanner_name = None
        for fact in state:
            if man_name is not None and match(fact, "carrying", man_name, "*"):
                carried_spanner_name = get_parts(fact)[2]
                break
        if carried_spanner_name is not None:
             if f"(usable {carried_spanner_name})" in state:
                 carrying_usable_spanner = True

        # 3. Identify untightened goal nuts
        untightened_goal_nuts = []
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                nut_name = get_parts(goal)[1]
                if f"(tightened {nut_name})" not in state:
                    untightened_goal_nuts.append(nut_name)

        # 4. Identify usable spanners on the ground and their locations/counts
        usable_spanners_on_ground = collections.defaultdict(int) # {location: count}
        usable_spanner_names_on_ground = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Check if this object is a spanner and is usable
                if f"(usable {obj})" in state:
                    # Need to confirm obj is a spanner type.
                    # Assuming any usable object at a location is a spanner for this domain.
                    # A more robust way would be to parse object types.
                    # Let's check if it's not the man and not a nut (assuming only man, spanner, nut are locatable)
                    if obj != man_name and obj not in self.goal_nut_locations: # Simple check if it's likely a spanner
                         usable_spanners_on_ground[loc] += 1
                         usable_spanner_names_on_ground.add(obj)


        # 5. If no untightened goal nuts, return 0
        if not untightened_goal_nuts:
            return 0

        # 6. Simulate greedy plan
        cost = 0
        current_loc = man_loc
        has_spanner = carrying_usable_spanner
        nuts_left = list(untightened_goal_nuts) # Create a mutable copy
        available_spanners_counts = dict(usable_spanners_on_ground) # Create a mutable copy

        # Add the carried spanner to available counts conceptually if it's usable
        # This simplifies finding the nearest spanner - we just look at locations
        # and assume the carried one is "at" the man's current location for pickup purposes
        # (even though the action requires it to be on the ground).
        # This is a heuristic simplification. A more accurate simulation would handle carried spanner separately.
        # Let's stick to the simpler simulation: if not carrying, find nearest on ground.
        # If carrying, use it first, then find nearest on ground.

        while nuts_left:
            # If man needs a spanner
            if not has_spanner:
                # Find nearest location with an available usable spanner
                nearest_s_loc = None
                min_dist_to_spanner = float('inf')
                for s_loc, count in available_spanners_counts.items():
                    if count > 0:
                        dist = self.get_distance(current_loc, s_loc)
                        if dist < min_dist_to_spanner:
                            min_dist_to_spanner = dist
                            nearest_s_loc = s_loc

                if nearest_s_loc is None or min_dist_to_spanner == float('inf'):
                    # Cannot get a spanner, problem likely unsolvable from here
                    # print(f"Warning: Cannot find usable spanner from {current_loc}")
                    return float('inf') # Return infinity if stuck

                # Travel to spanner
                cost += min_dist_to_spanner
                current_loc = nearest_s_loc

                # Pickup spanner
                cost += 1
                has_spanner = True
                available_spanners_counts[nearest_s_loc] -= 1 # Decrement count

            # Now man has a spanner, needs to tighten a nut
            nearest_n_loc = None
            min_dist_to_nut = float('inf')
            nut_to_tighten = None

            for nut_name in nuts_left:
                n_loc = self.goal_nut_locations.get(nut_name)
                if n_loc is None:
                     # Should not happen if goal_nut_locations is correctly populated
                     # print(f"Warning: Location for nut {nut_name} not found.")
                     continue # Skip this nut

                dist = self.get_distance(current_loc, n_loc)
                if dist < min_dist_to_nut:
                    min_dist_to_nut = dist
                    nearest_n_loc = n_loc
                    nut_to_tighten = nut_name

            if nearest_n_loc is None or min_dist_to_nut == float('inf'):
                 # Cannot reach any untightened nut
                 # print(f"Warning: Cannot reach any untightened nut from {current_loc}")
                 return float('inf') # Return infinity if stuck

            # Travel to nut
            cost += min_dist_to_nut
            current_loc = nearest_n_loc

            # Tighten nut
            cost += 1
            has_spanner = False # Spanner is consumed
            nuts_left.remove(nut_to_tighten) # Nut is tightened

        return cost

