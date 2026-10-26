from heuristics.heuristic_base import Heuristic

# Helper function to parse facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the cost to tighten all loose goal nuts.
    It considers the number of loose goal nuts, the travel cost for the man
    to reach a nut location, and the cost to acquire a usable spanner if needed.

    # Assumptions
    - The goal is always to tighten a set of nuts.
    - The location graph defined by 'link' predicates is connected enough
      to reach necessary locations (nuts, spanners) from the initial man location.
    - Usable spanners exist somewhere if needed for a solvable problem.
    - Spanners become unusable after one use and cannot be repaired or dropped.
    - The man object is named 'bob'.
    - Objects appearing as arguments to 'usable' or as the carried item in 'carrying'
      are spanners.

    # Heuristic Initialization
    - Extract the goal conditions (which nuts need to be tightened).
    - Build the location graph from 'link' static facts to compute distances.
    - Identify all location objects mentioned in links.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Identify all goal nuts that are currently loose in the state. If none are loose, the heuristic is 0 (goal state).
    2. Initialize the heuristic value with the count of loose goal nuts. This represents the minimum number of 'tighten_nut' actions required.
    3. Find the man's current location from the state. If the man's location is not found, return infinity (invalid state).
    4. Find the locations of all loose goal nuts from the state.
    5. Calculate the shortest distance from the man's current location to all other locations using Breadth-First Search (BFS) on the location graph built from static 'link' facts.
    6. Find the minimum distance from the man's current location to any of the loose nut locations. Add this minimum distance to the heuristic. If no loose nut locations are reachable, return infinity.
    7. Check if the man is currently carrying a usable spanner by examining the state facts.
    8. If the man is NOT carrying a usable spanner:
       a. Identify all usable spanners currently on the ground by examining the state facts ('at' and 'usable' predicates, and not 'carrying').
       b. Find the locations of these usable spanners.
       c. If there are no usable spanners on the ground, and the man doesn't have one, the state is likely unsolvable. Return infinity.
       d. Calculate the shortest distance from the man's current location to the closest usable spanner location on the ground using the precomputed BFS distances.
       e. Add this minimum distance plus 1 (for the 'pickup_spanner' action) to the heuristic.
    9. Return the total calculated heuristic value.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal conditions, building the
        location graph, and identifying all locations.
        """
        self.goals = task.goals  # Goal conditions
        self.static_facts = task.static # Static facts

        # Identify all goal nuts from goal conditions
        self.goal_nuts = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == "tightened" and len(parts) > 1:
                self.goal_nuts.add(parts[1])

        # Build the location graph from 'link' facts
        self.graph = {}
        self.all_locations = set()

        # Collect all locations mentioned in link facts
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) > 2:
                loc1, loc2 = parts[1], parts[2]
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Initialize graph adjacency lists
        for loc in self.all_locations:
            self.graph[loc] = set() # Use set for neighbors

        # Add links to the graph
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] == "link" and len(parts) > 2:
                loc1, loc2 = parts[1], parts[2]
                self.graph[loc1].add(loc2)
                self.graph[loc2].add(loc1) # Links are bidirectional

    def _bfs(self, start_location):
        """
        Performs BFS starting from start_location to find distances to all reachable locations.
        Returns a dictionary {location: distance}.
        Handles cases where start_location might not be in the pre-built graph.
        """
        distances = {loc: float('inf') for loc in self.all_locations}

        # If start_location is not in the graph built from links, add it to distances
        # but it will be isolated unless connected by links.
        if start_location not in self.graph:
             distances[start_location] = 0
             # No need to add to graph if we only care about distances from start_location
             # and it has no links.
        else:
             distances[start_location] = 0
             queue = [start_location]
             visited = {start_location}

             while queue:
                 current_loc = queue.pop(0) # Dequeue

                 # Ensure current_loc is a valid key in the graph (should be if added from all_locations)
                 if current_loc in self.graph:
                     for neighbor in self.graph[current_loc]:
                         if neighbor not in visited:
                             visited.add(neighbor)
                             distances[neighbor] = distances[current_loc] + 1
                             queue.append(neighbor)

        return distances

    def __call__(self, node):
        """
        Compute the heuristic value for the given state.
        """
        state = node.state

        # 1. Identify loose goal nuts and their locations
        loose_goal_nuts = {} # {nut_name: location}
        man_location = None
        man_carrying_usable_spanner = False
        usable_spanner_locations_on_ground = set() # {location}

        # Collect potential spanner names from state facts
        potential_spanners = set()
        for fact in state:
            parts = get_parts(fact)
            if len(parts) > 1:
                if parts[0] == "usable":
                    potential_spanners.add(parts[1])
                elif parts[0] == "carrying" and len(parts) > 2:
                     potential_spanners.add(parts[2])

        # Parse state facts to find locations and status
        for fact in state:
            parts = get_parts(fact)
            if not parts: continue # Skip empty facts

            predicate = parts[0]

            if predicate == "at" and len(parts) > 2:
                obj, loc = parts[1], parts[2]
                if obj == "bob": # Assuming 'bob' is the man
                    man_location = loc
                elif obj in self.goal_nuts:
                     # This fact tells us where a goal nut is, but we need to check if it's loose later
                     pass
                elif obj in potential_spanners: # Check for spanners on ground
                     spanner_name = obj
                     is_usable = ('(usable ' + spanner_name + ')') in state
                     is_carried = ('(carrying bob ' + spanner_name + ')') in state # Assuming 'bob' is the man
                     if is_usable and not is_carried:
                         usable_spanner_locations_on_ground.add(loc)

            elif predicate == "loose" and len(parts) > 1:
                 nut_name = parts[1]
                 if nut_name in self.goal_nuts:
                     # Find its location from the state
                     nut_location = None
                     for other_fact in state:
                         other_parts = get_parts(other_fact)
                         if other_parts and other_parts[0] == "at" and len(other_parts) > 2 and other_parts[1] == nut_name:
                             nut_location = other_parts[2]
                             break
                     if nut_location:
                         loose_goal_nuts[nut_name] = nut_location
                     # else: nut is loose but not at any location? Problematic state.

            elif predicate == "carrying" and len(parts) > 2:
                 carrier, item = parts[1], parts[2]
                 if carrier == "bob": # Assuming 'bob' is the man
                     # Check if the carried item is a usable spanner
                     if item in potential_spanners:
                         if ('(usable ' + item + ')') in state:
                             man_carrying_usable_spanner = True


        # 1. Check if goal is reached (all goal nuts are tightened)
        # The goal is reached if there are no loose goal nuts.
        if not loose_goal_nuts:
            return 0

        # Ensure man_location was found (should always be the case in valid states)
        if man_location is None:
             # This indicates a problem with the state representation or domain.
             # A man must be somewhere. Return infinity as unsolvable.
             return float('inf')

        # Compute distances from the man's current location using BFS
        dist_from_man = self._bfs(man_location)

        # 2. Initialize heuristic with number of loose nuts (tighten actions)
        h = len(loose_goal_nuts)

        # 3. Add travel cost for the man to reach the closest loose nut location
        loose_nut_locations = set(loose_goal_nuts.values())
        min_dist_to_any_nut_loc = float('inf')

        # It's possible to have loose nuts without locations if the state is malformed,
        # but assuming valid states, loose_goal_nuts implies locations were found.
        # If loose_goal_nuts is not empty, loose_nut_locations should not be empty.
        # However, check reachability.
        found_reachable_nut_loc = False
        for loc in loose_nut_locations:
             if loc in dist_from_man and dist_from_man[loc] != float('inf'):
                 min_dist_to_any_nut_loc = min(min_dist_to_any_nut_loc, dist_from_man[loc])
                 found_reachable_nut_loc = True

        # If there are loose nuts but none are reachable from the man's current location
        if not found_reachable_nut_loc:
             return float('inf') # Unsolvable from this state

        h += min_dist_to_any_nut_loc

        # 4. Add cost for getting a usable spanner if needed
        if not man_carrying_usable_spanner:
            min_dist_to_spanner = float('inf')

            # Consider usable spanners on the ground
            found_reachable_spanner_loc = False
            for loc in usable_spanner_locations_on_ground:
                 if loc in dist_from_man and dist_from_man[loc] != float('inf'):
                     min_dist_to_spanner = min(min_dist_to_spanner, dist_from_man[loc])
                     found_reachable_spanner_loc = True

            # If no usable spanners are reachable on the ground and man doesn't have one
            if not found_reachable_spanner_loc:
                 # Unsolvable state (no way to get a usable spanner)
                 return float('inf')

            # Add cost to travel to the closest reachable usable spanner and pick it up
            h += min_dist_to_spanner + 1 # +1 for pickup action

        return h
