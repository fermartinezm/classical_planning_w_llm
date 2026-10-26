from fnmatch import fnmatch
from collections import deque
from heuristics.heuristic_base import Heuristic

# Helper functions (copied from example heuristic code)
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
    # Use zip to handle patterns shorter than fact parts
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions (tighten, pickup, walk) required
    to tighten all loose nuts. It calculates the cost for each loose nut independently
    and sums them up. The cost for a single nut includes the tighten action plus
    the estimated minimum cost to get the man to the nut's location carrying a
    usable spanner.

    # Assumptions
    - There is only one man.
    - Nuts are static (do not change location).
    - Spanners become unusable after one use and cannot become usable again.
    - Links between locations are bidirectional.
    - The problem is solvable (enough spanners exist). Unsolvable states return a large value.

    # Heuristic Initialization
    - Builds a graph of locations based on `link` facts.
    - Computes all-pairs shortest path distances between locations using BFS.
    - Stores the goal locations for each nut (nuts are static, so their location is fixed).

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1. Precomputation (`__init__`):
       - Parse `link` facts from static information to build the location graph (adjacency list).
       - Perform Breadth-First Search (BFS) starting from every location found in the graph
         to compute shortest path distances between all pairs of locations. Store these distances.
       - Identify all goal nuts from the task's goal conditions (facts like `(tightened nut1)`).
       - Find the fixed location for each goal nut by looking at the initial state facts
         (since nuts are static, their initial location is their only location).

    2. Heuristic Calculation (`__call__`):
       - Identify the current state facts.
       - Identify the man's name: Look for the object involved in any `carrying` fact. If none,
         look for the single object at a location that is not a known spanner or nut.
         If the man cannot be uniquely identified, return a large value.
       - Find the man's current location using an `at` fact. If the man is not at any
         location, return a large value.
       - Identify all nuts that are currently `loose` from the state facts.
       - If there are no loose nuts, the goal is reached, return 0.
       - Identify all spanners that are currently `usable` from the state facts.
       - Identify which usable spanners the man is currently `carrying`.
       - Identify which usable spanners are currently `at` a location (and not carried).
       - Check if the total number of usable spanners (carried + at locations) is
         less than the number of loose nuts. If so, the problem is unsolvable from
         this state, return a large value (representing infinity).
       - Initialize the total heuristic cost to 0.
       - For each loose nut:
         a. Add 1 to the cost for the `tighten_nut` action required for this nut.
         b. Retrieve the fixed location of this nut (found during initialization).
         c. Estimate the minimum cost to get the man to this nut's location while
            carrying a usable spanner. This cost is the minimum of two options:
            i.  Using a spanner the man is already carrying: The cost is the shortest
                distance from the man's current location to the nut's location. This
                option is only possible if the man is currently carrying at least one
                usable spanner.
            ii. Picking up a usable spanner from a location: Find the usable spanner
                at a location that minimizes the travel cost. The cost for a spanner
                at location L_S is: distance(man_loc, L_S) + 1 (pickup action) +
                distance(L_S, nut_loc). This option is only possible if there are
                usable spanners available at locations.
         d. Add the minimum cost calculated in step 6c to the total heuristic cost
            for this nut. If neither option is possible (e.g., no usable spanners
            anywhere reachable), return a large value (unsolvable).
       - The final heuristic value is the sum of the costs calculated for each loose nut.
         This additive approach overestimates the cost because travel can be shared
         between tasks for different nuts, but it provides a reasonable estimate
         for guiding a greedy search.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals = task.goals
        static_facts = task.static
        initial_state = task.initial_state # Need initial state to find nut locations

        # Build location graph
        self.location_graph = {}
        self.all_locations = set()
        for fact in static_facts:
            if match(fact, "link", "*", "*"):
                _, loc1, loc2 = get_parts(fact)
                self.location_graph.setdefault(loc1, set()).add(loc2)
                self.location_graph.setdefault(loc2, set()).add(loc1) # Links are bidirectional
                self.all_locations.add(loc1)
                self.all_locations.add(loc2)

        # Compute all-pairs shortest paths using BFS
        self.distances = {}
        for start_node in self.all_locations:
            q = deque([(start_node, 0)])
            visited = {start_node}
            self.distances[(start_node, start_node)] = 0

            while q:
                current_loc, dist = q.popleft()

                if current_loc in self.location_graph:
                    for neighbor in self.location_graph[current_loc]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            self.distances[(start_node, neighbor)] = dist + 1
                            q.append((neighbor, dist + 1))

        # Store goal nuts and their locations (nuts are static)
        self.goal_nuts = {}
        # Find all nut objects mentioned in the initial state or goals
        all_nuts_in_problem = set()
        for goal in self.goals:
             if match(goal, "tightened", "*"):
                 all_nuts_in_problem.add(get_parts(goal)[1])
        # Also consider nuts mentioned in the initial state, even if not in goals (though less common)
        for fact in initial_state:
             if match(fact, "at", "*", "*"):
                 obj = get_parts(fact)[1]
                 # Simple heuristic: check if name contains "nut"
                 if "nut" in obj.lower():
                     all_nuts_in_problem.add(obj)


        for nut_name in all_nuts_in_problem:
             # Find the location of this nut in the initial state (nuts are static)
             nut_location = None
             # Look for the nut's location in the initial state
             for fact in initial_state:
                 if match(fact, "at", nut_name, "*"):
                     nut_location = get_parts(fact)[2]
                     break
             if nut_location:
                 self.goal_nuts[nut_name] = nut_location
             else:
                 # This shouldn't happen in valid problems, but handle defensively
                 print(f"Warning: Could not find initial location for nut {nut_name}")


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Find man's name and location
        man_name = None
        # Find object involved in 'carrying' - must be the man
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                man_name = get_parts(fact)[1]
                break

        # If not found by carrying, find the single locatable that isn't a known nut or spanner
        if man_name is None:
            potential_men = set()
            all_spanners_in_state = {get_parts(fact)[1] for fact in state if match(fact, "*spanner*", "*")}
            all_nuts_in_state = {get_parts(fact)[1] for fact in state if match(fact, "*nut*", "*")}

            for fact in state:
                if match(fact, "at", "*", "*"):
                    obj = get_parts(fact)[1]
                    # Check if the object is not a spanner or nut based on name heuristic
                    if obj not in all_spanners_in_state and obj not in all_nuts_in_state:
                         potential_men.add(obj)

            if len(potential_men) == 1:
                man_name = list(potential_men)[0]
            # If len(potential_men) is 0 or > 1, man_name remains None

        if man_name is None:
            # Cannot identify the man, state is likely malformed for this heuristic
            return 1000000 # Indicate unsolvable or error

        man_location = None
        for fact in state:
            if match(fact, "at", man_name, "*"):
                man_location = get_parts(fact)[2]
                break

        if man_location is None:
            # Man is not at any location? State is inconsistent.
            return 1000000 # Indicate unsolvable or error

        # 2. Find loose nuts
        loose_nuts = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*")}

        # Goal reached if no loose nuts
        if not loose_nuts:
            return 0

        # 3. Find usable spanners
        usable_spanners = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*")}

        # 4. Find spanners carried by man
        spanners_carried = {get_parts(fact)[2] for fact in state if match(fact, "carrying", man_name, "*")}
        usable_spanners_carried = {s for s in spanners_carried if s in usable_spanners}

        # 5. Find usable spanners at locations
        usable_spanners_at_loc = {}
        for fact in state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:3]
                 if obj in usable_spanners and obj not in spanners_carried:
                     usable_spanners_at_loc[obj] = loc

        num_loose = len(loose_nuts)
        num_usable_carried = len(usable_spanners_carried)
        num_usable_at_loc = len(usable_spanners_at_loc)

        # Check solvability based on total usable spanners
        if num_loose > num_usable_carried + num_usable_at_loc:
             return 1000000 # Not enough usable spanners in total

        # Heuristic calculation: Sum of costs for each loose nut independently
        total_cost = 0

        for nut in loose_nuts:
            nut_location = self.goal_nuts.get(nut)
            if nut_location is None:
                # Location of a loose nut not found in goal_nuts (shouldn't happen)
                 return 1000000

            # Cost for this nut = 1 (tighten) + cost to get man+spanner to nut_location

            cost_to_get_man_spanner = float('inf')

            # Option 1: Use a carried usable spanner
            # Cost is just walking to the nut
            dist_via_carried = self.distances.get((man_location, nut_location), float('inf'))
            if num_usable_carried > 0 and dist_via_carried != float('inf'):
                 cost_to_get_man_spanner = min(cost_to_get_man_spanner, dist_via_carried)

            # Option 2: Pick up a usable spanner at a location
            if num_usable_at_loc > 0:
                # Find the minimum cost to get a spanner from a location and bring it to the nut
                min_pickup_travel_cost = float('inf')
                for spanner, spanner_loc in usable_spanners_at_loc.items():
                     # Cost = walk to spanner + pickup + walk to nut
                     dist_to_spanner = self.distances.get((man_location, spanner_loc), float('inf'))
                     dist_spanner_to_nut = self.distances.get((spanner_loc, nut_location), float('inf'))
                     if dist_to_spanner != float('inf') and dist_spanner_to_nut != float('inf'):
                         min_pickup_travel_cost = min(min_pickup_travel_cost, dist_to_spanner + 1 + dist_spanner_to_nut)

                cost_to_get_man_spanner = min(cost_to_get_man_spanner, min_pickup_travel_cost)


            if cost_to_get_man_spanner == float('inf'):
                # Cannot reach nut or any usable spanner location from man's current location
                return 1000000 # Unsolvable

            # Total cost for this nut = 1 (tighten) + estimated travel/pickup cost
            total_cost += 1 + cost_to_get_man_spanner

        return total_cost
