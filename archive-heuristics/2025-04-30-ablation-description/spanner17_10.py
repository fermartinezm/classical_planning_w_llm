from fnmatch import fnmatch
# Assuming Heuristic base class is available in heuristics.heuristic_base
# from heuristics.heuristic_base import Heuristic

# Define a dummy Heuristic base class if not running in the specific environment
# This allows the code to be runnable standalone for testing syntax/logic
try:
    from heuristics.heuristic_base import Heuristic
except ImportError:
    class Heuristic:
        def __init__(self, task):
            pass
        def __call__(self, node):
            raise NotImplementedError
        # Add dummy methods if the base class has abstract methods
        # For this problem, the base class only has __init__ and __call__


def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Handle potential empty string or malformed fact string
    if not fact or not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
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
    This heuristic estimates the minimum number of actions required to tighten all
    goal nuts. It considers the steps needed for each loose goal nut: moving the
    man to the nut's location, acquiring a usable spanner if not already carrying
    one, and performing the tighten action. It processes the loose nuts sequentially,
    updating the man's location and spanner status after each estimated tightening
    action.

    # Assumptions
    - There is exactly one man object in the domain.
    - Nuts' locations are static.
    - A man can carry multiple spanners.
    - A spanner becomes unusable after tightening one nut, but remains carried.
    - There is no action to drop a spanner or make an unusable spanner usable again.
    - Solvable problems guarantee enough usable spanners exist initially (on the ground or carried)
      to tighten all goal nuts.
    - The cost of walking between any two distinct locations the man needs to visit
      (current, nut, spanner) is estimated as 1 action. This simplifies complex pathfinding.

    # Heuristic Initialization
    - Identify the man's name by finding the unique object of type 'man'.
    - Extract the goal nuts that need to be tightened.
    - Extract the static locations of all nuts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic estimates the cost by considering the actions needed to tighten
    each loose goal nut in sequence.

    1.  **Identify Current State:** Get the man's current location, the set of
        spanners he is carrying, and the set of usable spanners on the ground.
    2.  **Identify Loose Goal Nuts:** Determine which nuts specified in the goal
        are currently in a `(loose ...)` state. If none are loose, the goal is met,
        and the heuristic is 0.
    3.  **Initialize Cost and State Variables:** Set the total cost to 0. Keep track
        of the man's estimated current location, the set of usable spanners he is
        estimated to be carrying, and the set of available usable spanners on the ground.
    4.  **Iterate Through Loose Nuts:** For each loose goal nut (processed in an
        arbitrary but fixed order, e.g., as they appear in the goal list):
        a.  **Move to Nut Location:** If the man is not currently at the nut's location,
            add 1 to the total cost and update the man's estimated location to the nut's location.
        b.  **Acquire Usable Spanner (if needed at location):** If the man does not
            have any usable spanners among those he is currently estimated to be carrying:
            i.  Find an available usable spanner on the ground. If none are available,
                return a very large number (indicating likely unsolvability).
            ii. Add 1 to the total cost for the `pickup_spanner` action.
            iii. If the man's current estimated location is different from the spanner's
                 location, add 1 to the total cost and update the man's estimated
                 location to the spanner's location.
            iv. Add the picked-up spanner to the set of usable spanners the man is carrying
                and remove it from the set of available ground spanners.
            v.  If the man's current estimated location (where he picked up the spanner)
                is different from the nut's location, add 1 to the total cost and
                update the man's estimated location back to the nut's location.
        c.  **Use Usable Spanner:** If the man has multiple usable spanners carried,
            select one to use for this nut. Remove it from the set of usable spanners
            the man is carrying. (The spanner remains carried but is no longer usable).
        d.  **Tighten Nut:** Add 1 to the total cost for the `tighten_nut` action.
    5.  **Return Total Cost:** The accumulated cost is the heuristic estimate.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        super().__init__(task)
        self.goals = task.goals

        # Identify goal nuts
        self.goal_nuts = {get_parts(g)[1] for g in self.goals if get_parts(g)[0] == 'tightened'}

        # Extract static nut locations
        self.nut_locations = {}
        # Nuts' locations are typically static, found in initial state or static facts
        # Check initial state and static facts for nut locations
        all_static_facts = set(task.initial_state) | set(task.static)
        for fact in all_static_facts:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 if obj in self.goal_nuts:
                     self.nut_locations[obj] = loc

        # Identify the man's name
        # Find objects that are nuts or spanners by looking at predicates
        # they appear in (loose, tightened, usable, carrying).
        nuts_and_spanners = set()
        for fact in task.initial_state:
            parts = get_parts(fact)
            if parts:
                predicate = parts[0]
                if predicate in ['loose', 'tightened']: # It's a nut
                    nuts_and_spanners.add(parts[1])
                elif predicate in ['usable']: # It's a spanner
                     nuts_and_spanners.add(parts[1])
                elif predicate in ['carrying']: # (carrying man spanner)
                     # The second argument is the spanner
                     if len(parts) > 2:
                         nuts_and_spanners.add(parts[2])


        man_candidates = set()
        for fact in task.initial_state:
             if match(fact, "at", "*", "*"):
                 obj, loc = get_parts(fact)[1:]
                 # The man is the object at a location that is not a known nut or spanner
                 if obj not in nuts_and_spanners:
                     man_candidates.add(obj)

        # Assume there is exactly one man
        if len(man_candidates) != 1:
             # This heuristic assumes a single man. Handle error or pick one?
             # For this problem, let's assume the first candidate is the man or raise error
             if not man_candidates:
                 # Fallback: Check objects list if available (Task object doesn't provide it directly)
                 # or assume a default name like 'bob' if examples consistently use it.
                 # Given the examples, 'bob' is a safe assumption if man_candidates is empty.
                 # A more robust parser would get object types from the PDDL problem file.
                 # Let's assume 'bob' if no man is found via predicates.
                 # print("Warning: Could not identify the man object via predicates. Assuming 'bob'.")
                 self.man_name = 'bob' # Default assumption based on examples
             else:
                 # print(f"Warning: Found {len(man_candidates)} man candidates. Assuming the first one: {next(iter(man_candidates))}")
                 self.man_name = next(iter(man_candidates))
        else:
             self.man_name = next(iter(man_candidates))


    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify Current State
        current_man_location = None
        for fact in state:
            if match(fact, "at", self.man_name, "*"):
                current_man_location = get_parts(fact)[2]
                break # Assuming man is at only one location

        carried_spanners = {get_parts(fact)[2] for fact in state if match(fact, "carrying", self.man_name, "*")}
        usable_carried_spanners = {S for S in carried_spanners if f"(usable {S})" in state}

        current_ground_spanners = set() # Store as (spanner_name, location)
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Check if it's a usable spanner on the ground (not carried by man)
                if obj not in carried_spanners and f"(usable {obj})" in state:
                     current_ground_spanners.add((obj, loc))


        # 2. Identify Loose Goal Nuts
        loose_goal_nuts = [N for N in self.goal_nuts if (f"(loose {N})") in state]

        if not loose_goal_nuts:
            return 0 # Goal reached

        # 3. Initialize Cost and State Variables
        total_cost = 0
        estimated_man_location = current_man_location
        estimated_usable_carried_spanners = set(usable_carried_spanners) # Copy the set
        available_ground_spanners = set(current_ground_spanners) # Copy the set

        # 4. Iterate Through Loose Nuts (in the order they appear in self.goal_nuts)
        # We use the order from self.goal_nuts to ensure a fixed processing order
        # Filter self.goal_nuts to only include those that are currently loose
        ordered_loose_goal_nuts = [N for N in self.goal_nuts if N in loose_goal_nuts]

        for nut_name in ordered_loose_goal_nuts:
            nut_location = self.nut_locations.get(nut_name)
            if nut_location is None:
                 # Should not happen in valid problems if nut_locations was built correctly
                 return 1000000 # Indicate problem with state/setup

            # a. Move to Nut Location
            if estimated_man_location != nut_location:
                total_cost += 1 # Walk action
                estimated_man_location = nut_location

            # b. Acquire Usable Spanner (if needed at location)
            if not estimated_usable_carried_spanners:
                # Man needs to get a usable spanner from the ground.
                if not available_ground_spanners:
                    # No usable spanners left on the ground and man isn't carrying one
                    # Problem likely unsolvable from this state
                    return 1000000 # Indicate very high cost / unsolvable

                # Find any available usable spanner on the ground
                # We just pick one arbitrarily (e.g., the first one in the set)
                spanner_needed, spanner_location = next(iter(available_ground_spanners))

                # Cost to move to spanner location if not already there
                if estimated_man_location != spanner_location:
                    total_cost += 1 # Walk action
                    estimated_man_location = spanner_location

                # Cost to pickup spanner
                total_cost += 1 # Pickup action

                # Remove the spanner from available ground spanners
                available_ground_spanners.remove((spanner_needed, spanner_location))

                # Add the spanner to his usable carried set
                estimated_usable_carried_spanners.add(spanner_needed)

                # Cost to move from spanner location back to nut location if different
                if estimated_man_location != nut_location:
                     total_cost += 1 # Walk action
                     estimated_man_location = nut_location

            # c. Use Usable Spanner
            # Man has at least one usable spanner carried (either initially or just picked up)
            spanner_to_use = next(iter(estimated_usable_carried_spanners)) # Pick any usable carried spanner
            estimated_usable_carried_spanners.remove(spanner_to_use) # It becomes unusable

            # d. Tighten Nut
            total_cost += 1 # Tighten action

        # 5. Return Total Cost
        return total_cost
