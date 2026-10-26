from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic
from collections import deque # Using deque for more efficient BFS queue

def get_parts(fact):
    """Helper to split a PDDL fact string into predicate and arguments."""
    # Example: '(at obj loc)' -> ['at', 'obj', 'loc']
    return fact[1:-1].split()

def match(fact, *args):
    """Helper to check if a fact matches a pattern using fnmatch."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Define the heuristic class inheriting from Heuristic
# Uncomment the line below and remove the one after it when integrating
# class spannerHeuristic(Heuristic):
class spannerHeuristic:
    """
    Domain-dependent heuristic for the Spanner domain.

    Estimates the required number of actions to reach a goal state.
    This heuristic is non-admissible and designed to guide a greedy best-first search.

    Summary:
    The heuristic estimates the cost by summing three main components:
    1. The number of goal nuts that are currently loose (representing the minimum number of 'tighten_nut' actions needed).
    2. The shortest path distance (number of 'walk' actions) from the man's current location to the location of the closest loose goal nut.
    3. An additional cost (shortest path distance to a usable spanner + 1 for 'pickup_spanner') if the man is not currently carrying a usable spanner and there are loose nuts remaining.

    Assumptions:
    - Links between locations defined by the 'link' predicate are bidirectional.
    - Nuts are static objects; their location does not change from the initial state.
    - The man object can be identified (e.g., by appearing in 'at' or 'carrying' facts and not being a spanner or nut).
    - Usable spanners are consumed after one use ('tighten_nut' removes the 'usable' predicate for that spanner).
    - The heuristic estimates the cost for the *first* spanner acquisition trip if needed, and the walk to the *closest* nut, plus the count of remaining nuts. It does not perfectly model complex sequences of actions involving multiple spanners or visiting nuts in an optimal order.
    - Reachability between locations is determined solely by the 'link' facts in the static information.

    Heuristic Initialization:
    During initialization, the heuristic performs preprocessing steps:
    - Stores goal facts, static facts, and initial state.
    - Identifies all objects (man, nuts, spanners, locations) present in the problem definition by inspecting initial state, goals, and static facts.
    - Builds a graph representing the locations and the links between them based on 'link' facts. Assumes links are bidirectional.
    - Computes all-pairs shortest path distances between all reachable locations using Breadth-First Search (BFS).
    - Stores the static locations of all goal nuts by looking them up in the initial state.
    """

    def __init__(self, task):
        """
        Initializes the heuristic by parsing task information and precomputing distances.

        Args:
            task: The planning task object containing initial state, goals, operators, and static facts.
        """
        self.goals = task.goals
        self.static_facts = task.static
        self.initial_state = task.initial_state

        # Parse objects and types from initial state, goals, and static facts
        self.objects = {} # {object_name: type}
        self.man = None
        self.nuts = set()
        self.spanners = set()
        self.locations = set()

        # Collect all relevant facts to find all objects and locations
        all_relevant_facts = set(self.initial_state) | set(self.static_facts) | set(self.goals)

        for fact_str in all_relevant_facts:
            parts = get_parts(fact_str)
            if not parts: continue # Skip empty facts if any

            predicate = parts[0]
            args = parts[1:]

            # Attempt to identify objects and their types based on predicates
            if predicate == 'at':
                obj, loc = args
                # Simple inference: objects at locations are locatable or locations themselves
                self.objects[obj] = self.objects.get(obj, 'locatable') # Default to locatable
                self.locations.add(loc)
                self.objects[loc] = 'location'
            elif predicate == 'carrying':
                 m, s = args
                 self.man = m # Found the man
                 self.objects[m] = 'man'
                 self.spanners.add(s) # Found a spanner
                 self.objects[s] = 'spanner'
            elif predicate == 'loose' or predicate == 'tightened':
                 n = args[0]
                 self.nuts.add(n) # Found a nut
                 self.objects[n] = 'nut'
            elif predicate == 'usable':
                 s = args[0]
                 self.spanners.add(s) # Found a spanner
                 self.objects[s] = 'spanner'
            elif predicate == 'link':
                 l1, l2 = args
                 self.locations.add(l1) # Found locations
                 self.locations.add(l2)
                 self.objects[l1] = 'location'
                 self.objects[l2] = 'location'
            # Note: This parsing is heuristic. A proper parser would use PDDL type definitions.
            # We refine types based on predicate usage.

        # Refine types based on collected sets
        for obj in self.objects:
            if obj == self.man:
                self.objects[obj] = 'man'
            elif obj in self.nuts:
                self.objects[obj] = 'nut'
            elif obj in self.spanners:
                self.objects[obj] = 'spanner'
            elif obj in self.locations:
                 self.objects[obj] = 'location'
            # Any remaining 'locatable' could be other types if domain had them

        # Ensure man object was found
        if self.man is None:
             # Fallback: Assume the first object not identified as nut/spanner/location is the man
             for fact_str in self.initial_state:
                 parts = get_parts(fact_str)
                 if parts[0] == 'at':
                     obj = parts[1]
                     if obj not in self.nuts and obj not in self.spanners and obj not in self.locations:
                         self.man = obj
                         self.objects[obj] = 'man'
                         print(f"Warning: Man object inferred as '{self.man}'.")
                         break
             if self.man is None:
                  print("Error: Man object could not be identified.") # Should not happen in valid problems

        # Build location graph
        self.location_graph = {}
        for loc in self.locations:
             self.location_graph[loc] = set() # Initialize all known locations

        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts[0] == 'link':
                loc1, loc2 = parts[1], parts[2]
                # Only add links if both locations are known
                if loc1 in self.locations and loc2 in self.locations:
                    self.location_graph[loc1].add(loc2)
                    self.location_graph[loc2].add(loc1) # Assuming links are bidirectional

        # Compute all-pairs shortest paths
        self.distances = self._compute_all_pairs_shortest_paths()

        # Store goal nut locations (nuts are static)
        self.goal_nut_locations = {}
        goal_nuts_set = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == 'tightened':
                nut = parts[1]
                goal_nuts_set.add(nut)

        # Find initial location of goal nuts
        for nut in goal_nuts_set:
             found_loc = False
             for fact in self.initial_state:
                 p = get_parts(fact)
                 if p[0] == 'at' and p[1] == nut:
                     self.goal_nut_locations[nut] = p[2]
                     found_loc = True
                     break
             if not found_loc:
                 # This goal nut was not located in the initial state.
                 # This might indicate an unsolvable problem or a complex domain feature.
                 # For this heuristic, we assume goal nuts are statically located in initial state.
                 print(f"Warning: Goal nut {nut} location not found in initial state.")
                 # This nut will not be included in loose_goal_nuts if its location isn't known.


    def _compute_all_pairs_shortest_paths(self):
        """Computes shortest path distances between all pairs of locations using BFS."""
        distances = {}
        locations = list(self.location_graph.keys())
        for start_loc in locations:
            distances[start_loc] = self._bfs(start_loc)
        return distances

    def _bfs(self, start_loc):
        """Performs BFS from a start location to find distances to all reachable locations."""
        dist = {start_loc: 0}
        queue = deque([start_loc]) # Use deque for efficient queue operations
        visited = {start_loc}

        while queue:
            curr_loc = queue.popleft()

            # Ensure curr_loc is in graph (should be if from self.locations)
            if curr_loc in self.location_graph:
                for neighbor in self.location_graph[curr_loc]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        dist[neighbor] = dist[curr_loc] + 1
                        queue.append(neighbor)
        return dist


    def __call__(self, node):
        """
        Computes the domain-dependent heuristic value for a given state.

        Step-By-Step Thinking for Computing Heuristic:
        1. Identify which goal nuts are currently loose in the state by checking for the '(loose ?nut)' predicate for each goal nut.
        2. If no goal nuts are loose, the state is a goal state, return 0.
        3. Initialize heuristic value with the count of loose goal nuts (representing the minimum number of 'tighten_nut' actions required).
        4. Find the man's current location by searching for the '(at ?man ?location)' predicate in the state. If the man's location cannot be determined or is not in the precomputed distances, return infinity (indicating an unreachable or invalid state).
        5. Get the locations of all identified loose goal nuts using the precomputed goal nut locations.
        6. Calculate the minimum walk distance from the man's current location to any of the loose nut locations using the precomputed all-pairs shortest path distances. Add this minimum distance to the heuristic (representing the estimated cost to reach the first nut). If no loose nut location is reachable from the man's current location, return infinity.
        7. Determine if the man is currently carrying a usable spanner by checking for '(carrying ?man ?spanner)' and '(usable ?spanner)' predicates in the state.
        8. If there are loose nuts remaining (checked in step 2) AND the man is NOT currently carrying a usable spanner:
            a. Find all usable spanners that are currently located somewhere (not carried by the man) by checking for '(at ?spanner ?location)' and '(usable ?spanner)' predicates.
            b. Calculate the minimum walk distance from the man's current location to the location of any of these available usable spanners.
            c. If at least one usable spanner is found at a reachable location, add this minimum distance plus 1 (for the 'pickup_spanner' action) to the heuristic. This represents the estimated cost to acquire the first necessary tool.
            d. If no usable spanner is found anywhere (neither carried nor at a reachable location), return infinity (indicating an unsolvable state from this point).
        9. Return the final calculated heuristic value.

        Assumptions:
        - The heuristic is non-admissible.
        - It provides an estimate and does not guarantee finding the shortest plan.
        """
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = {
            nut for nut in self.goal_nut_locations.keys()
            if f'(loose {nut})' in state
        }

        # 2. If all goal nuts are tightened, heuristic is 0
        if not loose_goal_nuts:
            return 0

        heuristic = 0

        # 3. Initialize heuristic with cost for tighten actions
        # Each loose nut requires at least one tighten action
        heuristic += len(loose_goal_nuts)

        # 4. Find man's current location
        man_loc = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'at' and parts[1] == self.man:
                man_loc = parts[2]
                break

        # If man's location is unknown or not in our location graph, state is likely invalid/unreachable
        if man_loc is None or man_loc not in self.distances:
             return float('inf') # Indicate this state is likely not on a path to the goal

        # 5. Find locations of loose goal nuts and the closest one
        loose_nut_locations = {self.goal_nut_locations[nut] for nut in loose_goal_nuts}
        min_dist_to_nut = float('inf')
        can_reach_any_nut = False

        # Calculate minimum distance from man to any loose nut location
        if man_loc in self.distances: # Ensure man's location is in the distance map
            for nut_loc in loose_nut_locations:
                if nut_loc in self.distances[man_loc]: # Ensure nut location is reachable
                    min_dist_to_nut = min(min_dist_to_nut, self.distances[man_loc][nut_loc])
                    can_reach_any_nut = True

        # If no loose nut location is reachable from man's current location
        if not can_reach_any_nut:
            return float('inf') # Indicate this state is likely not on a path to the goal

        # 6. Add cost to reach the closest loose nut
        heuristic += min_dist_to_nut

        # 7. Check if man is carrying a usable spanner
        man_carrying_usable = False
        carried_spanners = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == 'carrying' and parts[1] == self.man:
                carried_spanners.add(parts[2])

        for spanner in carried_spanners:
            # Check if the carried spanner is usable
            if f'(usable {spanner})' in state:
                man_carrying_usable = True
                break # Found one usable carried spanner, that's enough for this part of heuristic

        # 8. If work is needed AND man is not carrying a usable spanner
        if len(loose_goal_nuts) > 0 and not man_carrying_usable:
            min_dist_to_spanner = float('inf')
            usable_spanner_available_somewhere = False

            # Find usable spanners not carried by man and their locations
            for spanner in self.spanners: # Iterate through all known spanners
                 # Check if spanner is usable and not carried by the man
                 if spanner not in carried_spanners and f'(usable {spanner})' in state:
                     # This spanner is usable and not carried. Find its current location.
                     spanner_loc = None
                     for fact in state:
                         parts = get_parts(fact)
                         if parts[0] == 'at' and parts[1] == spanner:
                             spanner_loc = parts[2]
                             break
                     # If spanner location found and reachable from man's location
                     if spanner_loc and man_loc in self.distances and spanner_loc in self.distances[man_loc]:
                         min_dist_to_spanner = min(min_dist_to_spanner, self.distances[man_loc][spanner_loc])
                         usable_spanner_available_somewhere = True

            # If a usable spanner is available somewhere reachable
            if usable_spanner_available_somewhere:
                # Add cost to get the closest usable spanner (walk + pickup)
                heuristic += min_dist_to_spanner + 1
            else:
                # Man needs a spanner, but none are available/usable anywhere reachable.
                # This state is likely unsolvable.
                return float('inf') # Indicate this state is likely not on a path to the goal

        # 9. Return the calculated heuristic value
        return heuristic

