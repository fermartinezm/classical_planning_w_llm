# Assuming Heuristic base class is available here
from heuristics.heuristic_base import Heuristic

# Helper functions (as in Logistics example)
from collections import deque
from fnmatch import fnmatch

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

# Heuristic class
class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Estimates the cost to tighten all goal nuts by summing:
    1. The number of loose goal nuts (each needs a tighten action).
    2. The number of spanners the man needs to pick up.
    3. An approximation of the travel cost to visit all required nut locations
       and spanner pickup locations.

    Assumes:
    - Each tighten action requires one usable spanner.
    - Spanners become unusable after one tighten action.
    - The man can carry multiple spanners.
    - Locations are connected by bidirectional links.
    - Shortest path distances are precomputed.
    - The man object can be identified.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts, locations, links,
        computing shortest paths, and identifying the man object.
        """
        self.task = task
        # Identify the names of the nuts that need to be tightened in the goal
        self.goal_nuts = {get_parts(goal)[1] for goal in task.goals if match(goal, "tightened", "*")}

        self.locations = set()
        self.links = set()

        # Collect all unique location names mentioned in initial state 'at' facts
        initial_at_facts = []
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 parts = get_parts(fact)
                 obj, loc = parts[1], parts[2]
                 self.locations.add(loc)
                 initial_at_facts.append((obj, loc))

        # Collect all unique location names and links from static facts
        for fact in task.static:
            if match(fact, "link", "*", "*"):
                parts = get_parts(fact)
                l1, l2 = parts[1], parts[2]
                self.links.add((l1, l2))
                self.locations.add(l1)
                self.locations.add(l2)

        # Compute all-pairs shortest paths between all identified locations
        self.dist = self._get_shortest_paths(list(self.locations), list(self.links))

        # Identify the man's name. This is heuristic based on available task info.
        self.man_name = None
        # 1. Try finding the object involved in an initial 'carrying' fact.
        for fact in task.initial_state:
            if match(fact, "carrying", "*", "*"):
                self.man_name = get_parts(fact)[1]
                break
        # 2. Fallback: Find the object in an initial 'at' fact that is not a known spanner or nut.
        #    This assumes there's only one man and he's distinct from spanners/nuts by predicates.
        if self.man_name is None:
             known_objects = set()
             # Add objects from initial state that are spanners (usable) or nuts (loose)
             for fact in task.initial_state:
                 parts = get_parts(fact)
                 if match(fact, "usable", "*") or match(fact, "loose", "*"):
                     known_objects.add(parts[1])
             # Add objects from goals that are nuts (tightened)
             for goal in task.goals:
                 if match(goal, "tightened", "*"):
                     known_objects.add(get_parts(goal)[1])

             for obj, loc in initial_at_facts:
                 if obj not in known_objects:
                     self.man_name = obj
                     break

        if self.man_name is None:
             # If man name still couldn't be identified, the heuristic might fail.
             # print("Warning: Could not identify the man object.")
             pass # Heuristic will return inf if man_loc cannot be found later


    def _get_shortest_paths(self, locations, links):
        """
        Calculates shortest path distances between all pairs of locations using BFS.
        Assumes links are bidirectional. Returns a dict of dicts: dist[start][end] = distance.
        Handles disconnected components (unreachable nodes won't appear in inner dict).
        """
        adj = {loc: set() for loc in locations}
        for l1, l2 in links:
            # Ensure linked locations are in our collected set before adding to adjacency list
            if l1 in adj and l2 in adj:
                adj[l1].add(l2)
                adj[l2].add(l1)
            # else: print(f"Warning: Link {l1}-{l2} involves location not found in initial state 'at' facts or other links.")


        dist = {}
        for start_node in locations:
            dist[start_node] = {}
            q = deque([(start_node, 0)])
            visited = {start_node}
            while q:
                current_node, d = q.popleft()
                dist[start_node][current_node] = d
                # Check if current_node is in adj (should be if from locations list)
                if current_node in adj:
                    for neighbor in adj[current_node]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            q.append((neighbor, d + 1))
        return dist

    def __call__(self, node):
        """
        Compute the domain-dependent heuristic estimate for the given state.
        """
        state = node.state

        # Parse relevant facts from the current state
        man_loc = None
        carried_spanners = set()
        usable_spanners = set()
        at_loc = {} # Map object name to its location
        loose_nuts = set()
        tightened_nuts = set()

        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "at":
                obj, loc = parts[1], parts[2]
                at_loc[obj] = loc
                # Find the man's current location using the name identified in __init__
                if obj == self.man_name:
                    man_loc = loc
            elif parts[0] == "carrying":
                # man_name = parts[1] # Should match self.man_name
                spanner_name = parts[2]
                carried_spanners.add(spanner_name)
            elif parts[0] == "usable":
                spanner_name = parts[1]
                usable_spanners.add(spanner_name)
            elif parts[0] == "loose":
                nut_name = parts[1]
                loose_nuts.add(nut_name)
            elif parts[0] == "tightened":
                nut_name = parts[1]
                tightened_nuts.add(nut_name)

        # If man's location couldn't be determined (e.g., man_name not found or no 'at' fact for man),
        # the state is likely invalid or unsolvable.
        if man_loc is None:
             return float('inf')

        # Identify loose nuts that are also goal nuts. These are the ones we need to tighten.
        loose_goal_nuts = loose_nuts.intersection(self.goal_nuts)
        num_loose_goal_nuts = len(loose_goal_nuts)

        # If there are no loose goal nuts, the goal is reached. Heuristic is 0.
        if num_loose_goal_nuts == 0:
            return 0

        # Identify usable spanners currently carried by the man.
        usable_carried = carried_spanners.intersection(usable_spanners)
        num_usable_carried = len(usable_carried)

        # Identify usable spanners currently on the ground.
        usable_ground_spanners = usable_spanners - carried_spanners
        # Map locations to the count of usable ground spanners at that location
        usable_ground_spanners_at_loc_count = {}
        for s in usable_ground_spanners:
            if s in at_loc: # Ensure the spanner has a location in the current state
                loc = at_loc[s]
                usable_ground_spanners_at_loc_count[loc] = usable_ground_spanners_at_loc_count.get(loc, 0) + 1

        # Count the total number of usable spanners available in the current state (carried or on ground).
        total_usable_spanners_in_state = num_usable_carried + len(usable_ground_spanners) # Count individual spanners

        # Check if there are enough usable spanners in the current state to tighten all goal nuts.
        # If not, the problem is unsolvable from this state.
        if num_loose_goal_nuts > total_usable_spanners_in_state:
             return float('inf')

        # Calculate heuristic components
        h = 0

        # 1. Cost for 'tighten_nut' actions: One action per loose goal nut.
        h += num_loose_goal_nuts

        # 2. Cost for 'pickup_spanner' actions: Need to pick up spanners if the man
        #    doesn't carry enough usable ones for the remaining nuts.
        spanners_to_pickup = max(0, num_loose_goal_nuts - num_usable_carried)
        h += spanners_to_pickup

        # 3. Travel cost: Estimate the cost for the man to reach all necessary locations.
        #    Necessary locations include:
        #    - The location of each loose goal nut.
        #    - Locations where the man needs to pick up spanners.

        # Locations of loose goal nuts
        nut_locs_to_visit = {at_loc[n] for n in loose_goal_nuts if n in at_loc}

        # Locations where spanners need to be picked up.
        # We need to pick up 'spanners_to_pickup' individual spanners.
        # We select locations containing usable ground spanners, prioritizing closer ones,
        # until we have identified enough spanners to cover the pickup requirement.
        spanner_locs_to_visit = set()
        spanners_accounted_for = 0
        # Sort locations containing usable ground spanners by distance from the man's current location
        sorted_spanner_locations = sorted(
             list(usable_ground_spanners_at_loc_count.keys()),
             key=lambda loc: self.dist.get(man_loc, {}).get(loc, float('inf'))
        )

        for loc in sorted_spanner_locations:
            if spanners_accounted_for < spanners_to_pickup:
                spanner_locs_to_visit.add(loc)
                spanners_accounted_for += usable_ground_spanners_at_loc_count[loc]
            else:
                break # Have identified enough locations to cover needed pickups

        # The set of all locations the man needs to visit (approximately)
        required_locs = nut_locs_to_visit.union(spanner_locs_to_visit)

        # Travel cost approximation: Sum of shortest path distances from the man's
        # current location to each required location. This is an overestimate
        # but simple and captures the increasing cost with distance and number of locations.
        travel_cost = 0
        # Check if man's location is in the graph and reachable from itself (always true if in graph)
        if man_loc not in self.dist:
             # Man's location is not in the graph of linked locations. Problem setup issue or unsolvable.
             return float('inf')

        for loc in required_locs:
            # Check if the required location is reachable from the man's current location
            if loc not in self.dist[man_loc]:
                 # Required location is unreachable. Problem is unsolvable from this state.
                 return float('inf')
            travel_cost += self.dist[man_loc][loc]

        h += travel_cost

        return h
