from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern with wildcards.

    Args:
        fact (str): PDDL fact string, e.g. "(at bob shed)".
        args (str): Pattern parts with optional wildcards.

    Returns:
        bool: True if all parts match, False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    Summary:
    ----------
    This heuristic estimates the number of actions required to tighten all nuts
    (i.e., achieve `(tightened ?n)` for all nuts in the goal) from the current state.
    It does so by computing, for each loose nut, an optimistic but informed count 
    of actions needed to pick up a usable spanner, move both spanner and man to the nut's
    location, and perform the tighten action. The final heuristic is the sum of these 
    minimal costs over all loose goal nuts.

    Assumptions:
    ------------
    - There is only one man.
    - Each tighten action requires the man to be at the nut's location,
      carrying a usable spanner.
    - Spanners become unusable after tightening a nut and must be replaced before
      tightening another nut.
    - The heuristic ignores spanner replacement sequencing and assumes spanners
      are always available nearby or currently carried.
    - Actions are weighted uniformly (each action counts as 1).
    - The domain graph defined by `(link loc1 loc2)` is undirected.
    - The heuristic is not guaranteed to be admissible but aims to be informative and efficient.

    Heuristic Initialization:
    --------------------------
    - Extract and store the undirected connectivity graph between locations from static `(link...)` facts.
    - Identify the unique man object from initial facts.
    - Extract all spanner objects and their initial usability.
    - Store the goal nuts that must be tightened.
  
    Step-By-Step Thinking for Computing Heuristic:
    -----------------------------------------------
    1. Identify all nuts in the goal and among them those that are still loose in the current state.
    2. Identify the current location of the man.
    3. For each spanner:
       a. Determine its current location if not carried.
       b. Determine if it is usable.
    4. For each loose nut:
       a. Identify the nut's current location.
       b. Estimate the minimal action cost to:
          - Obtain a usable spanner (either already carried or picking up the closest usable spanner).
          - Walk to the nut's location (from the man/spanner location).
          - Tighten the nut (1 action).
       c. The walking cost is computed as the shortest path distance (in number of links) through the connectivity graph.
       d. If man is already carrying a usable spanner, no pickup movement is needed; else the cost includes moving to and picking up a usable spanner.
    5. Since spanners become unusable after use, assume spanners are available and usable for each nut independently (optimistic).
    6. Sum the estimated minimal costs for all loose nuts.
    7. Return zero if no loose nuts remain.

    This heuristic balances accuracy and efficiency by using precomputed shortest distances and the current facts on spanner usability and positions.

    """

    def __init__(self, task):
        """
        Parses static facts and initial task information to build connectivity graphs,
        identify objects, and store the goal nuts.
        """
        self.goals = task.goals
        self.static = task.static

        # Build undirected location graph from (link loc1 loc2) facts.
        self.adjacency = {}
        for fact in self.static:
            if match(fact, "link", "*", "*"):
                _, l1, l2 = get_parts(fact)
                self.adjacency.setdefault(l1, set()).add(l2)
                self.adjacency.setdefault(l2, set()).add(l1)

        # Extract all location nodes
        self.locations = set(self.adjacency.keys())

        # Determine man objects by scanning initial facts for (at ?m ?loc) where ?m is man type
        # Since typing info is unavailable here, we identify the man by heuristics:
        # The domain only contains one man typically.
        # So, from initial state/info, find all objects appearing in (at ?m ?loc) with man semantics.
        # Assume the unique man: we find all objects in (at ?obj loc) where obj is not a location/spanner/nut,
        # likely just one man object.

        # First, collect all objects from initialized facts:
        self.man = None
        self.spanners = set()
        self.nuts = set()

        # Heuristics for object types from PDDL domain file:
        # man, nut, spanner, location
        # predicates: (at ?m - locatable ?l - location)
        # objects from domain are unknown, so we infer from facts:

        # Collect all atoms from static and goal for object identification
        # but since types not available, rely on fact naming conventions
        # We'll scan state facts for (carrying ?m ?s) to get the man, for example.

        # Instead, identify man by scanning static goals and initial
        # We look for any identifiers in the initial or static facts with 'man' in name or by usage:
        # We'll do a best attempt: find who carries spanners.

        # Store objects from facts with their recognized roles:
        # From static, goals, and initial state we will determine sets manually when heuristic is called,
        # but to accelerate, store goals nuts:

        # Extract nuts that must be tightened from the goal
        self.goal_nuts = set()
        for goal in self.goals:
            if match(goal, "tightened", "*"):
                self.goal_nuts.add(get_parts(goal)[1])

    def shortest_distance(self, start, goal):
        """
        Compute shortest path length between two locations using BFS.
        """
        if start == goal:
            return 0
        visited = {start}
        queue = [(start, 0)]
        while queue:
            loc, dist = queue.pop(0)
            for neighbor in self.adjacency.get(loc, []):
                if neighbor == goal:
                    return dist + 1
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))
        # If no path is found (disconnected graph), return a large distance
        return 1000

    def __call__(self, node):
        """
        Compute the heuristic estimate of remaining actions to tighten all loose nuts in the goal.

        The heuristic sums minimal action costs for each loose nut:
        - Moving to a usable spanner (if not already carrying one),
        - Picking it up,
        - Moving to the nut's location,
        - Tightening the nut (1 action).

        The man can carry only one spanner at a time. Each tightening makes the spanner unusable,
        so the heuristic assumes independent spanner availability for each nut (optimistic).

        Returns 0 for goal states.
        """
        state = node.state

        # If goal is reached, heuristic is 0
        if self.goals <= state:
            return 0

        # Extract current man location: (at ?man ?loc) where ?man is man object
        # Find the man by facts (carrying ?m ?s), if none use other (at ?m ?loc) where ?m not known as spanner/nut/location
        # Heuristic: find all man candidates based on carrying facts

        # Identify all men candidates:
        men_candidates = set()
        for fact in state:
            if match(fact, "carrying", "*", "*"):
                man_candidate = get_parts(fact)[1]
                men_candidates.add(man_candidate)
        # If none found, scan for (at ?obj ?loc) where ?obj not a nut or spanner
        if not men_candidates:
            for fact in state:
                if match(fact, "at", "*", "*"):
                    obj = get_parts(fact)[1]
                    if not obj.startswith("spanner") and not obj.startswith("nut"):
                        men_candidates.add(obj)
        # Take the one man found or the first from candidates
        if men_candidates:
            man = next(iter(men_candidates))
        else:
            # Fallback: just pick an object from at facts that is not spanner/nut/location
            man = None
            for fact in state:
                if match(fact, "at", "*", "*"):
                    obj = get_parts(fact)[1]
                    if not obj.startswith("spanner") and not obj.startswith("nut") and not obj in self.locations:
                        man = obj
                        break
            if man is None:
                # Fatal fallback: heuristic zero
                return 0

        # Find man's current location
        man_location = None
        for fact in state:
            if match(fact, "at", man, "*"):
                man_location = get_parts(fact)[2]
                break
        if man_location is None:
            # man location unknown, fallback heuristic zero
            return 0

        # Find all spanners, their location, and usability state
        spanner_locations = {}
        spanner_usable = set()
        spanner_carried = set()
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj = get_parts(fact)[1]
                loc = get_parts(fact)[2]
                if obj.startswith("spanner"):
                    spanner_locations[obj] = loc
            elif match(fact, "usable", "*"):
                spanner_usable.add(get_parts(fact)[1])
            elif match(fact, "carrying", man, "*"):
                spanner_carried.add(get_parts(fact)[2])

        # Track loose nuts that must be tightened
        loose_nuts = set()
        nut_locations = {}
        for fact in state:
            if match(fact, "loose", "*"):
                nut = get_parts(fact)[1]
                if nut in self.goal_nuts:
                    loose_nuts.add(nut)
            if match(fact, "at", "*", "*"):
                obj = get_parts(fact)[1]
                loc = get_parts(fact)[2]
                if obj.startswith("nut"):
                    nut_locations[obj] = loc

        # If no loose nuts remain, heuristic is 0
        if not loose_nuts:
            return 0

        # For each loose nut, compute cost to:
        # - get a usable spanner (if man carrying usable spanner, no pickup cost)
        # - walk to nut location (from spanner location or man's location)
        # - tighten nut (1 action)

        # Precompute shortest distances for efficiency
        # Distance between any two locations will be computed on demand using shortest_distance()

        # Determine if man is carrying a usable spanner
        man_carries_usable_spanner = False
        for s in spanner_carried:
            if s in spanner_usable:
                man_carries_usable_spanner = True
                carried_spanner = s
                break

        total_cost = 0

        # For each loose nut, estimate minimal cost independently (optimistic)
        for nut in loose_nuts:
            nut_loc = nut_locations.get(nut, None)
            if nut_loc is None:
                # Unknown nut location, skip or consider large cost
                total_cost += 10
                continue

            # Cost components:
            # 1) Pickup cost:
            #    If man has usable spanner, pickup cost = 0
            #    Else:
            #      Find minimal distance to get and pickup a usable spanner
            #      Cost to move to spanner location + 1 pickup action
            #      Then move from spanner location to nut location
            # 2) If man carries usable spanner:
            #      cost = walk distance man_location->nut_loc + 1 (tighten)
            # 3) Actions cost counting:
            #      walk actions = distance in links (1 per link)
            #      pickup_spanner = 1 action
            #      tighten_nut = 1 action

            if man_carries_usable_spanner:
                # Directly walk to nut and tighten
                walk_dist = self.shortest_distance(man_location, nut_loc)
                cost = walk_dist + 1  # tighten_nut
            else:
                # Find minimal cost to get usable spanner
                min_spanner_cost = 1000  # large number
                for spanner in spanner_usable:
                    # If spanner is currently carried by man, it would be covered above
                    # Get spanner location
                    sp_loc = spanner_locations.get(spanner, None)
                    if sp_loc is None:
                        continue
                    # Cost to walk man_location -> spanner + pick up (1) + walk spanner -> nut_loc
                    dist_to_sp = self.shortest_distance(man_location, sp_loc)
                    dist_sp_to_nut = self.shortest_distance(sp_loc, nut_loc)
                    total_spanner_cost = dist_to_sp + 1 + dist_sp_to_nut + 1  # +1 tighten
                    if total_spanner_cost < min_spanner_cost:
                        min_spanner_cost = total_spanner_cost
                cost = min_spanner_cost

            total_cost += cost

        # The heuristic value is the sum of the minimal estimated costs for all loose nuts
        return total_cost
