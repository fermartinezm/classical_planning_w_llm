from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

# Helper functions to parse PDDL facts
def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at obj1 loc1)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts matches the number of args, unless args has wildcards at the end
    if len(parts) < len(args) or not all(fnmatch(part, arg) for part, arg in zip(parts, args)):
        return False
    # If args is shorter than parts, check if remaining args are wildcards (not strictly needed by fnmatch but good practice)
    # For this domain, fact parts correspond directly to predicate and arguments, so simple zip is fine.
    return True


class spannerHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the Spanner domain.

    # Summary
    This heuristic estimates the number of actions required to tighten all
    loose nuts specified in the goal. It considers the number of nuts to tighten,
    the number of spanners that need to be picked up, and a simplified estimate
    of the movement cost for the man to visit relevant locations.

    # Assumptions
    - There is exactly one man agent.
    - Spanners are consumed after one use (become unusable).
    - Enough usable spanners exist in the problem instance to tighten all goal nuts.
    - The movement cost is approximated by the number of distinct relevant locations
      (nut locations and spanner pickup locations) the man needs to visit.

    # Heuristic Initialization
    - Extracts the set of nuts that must be tightened in the goal state from the task's goals.

    # Step-By-Step Thinking for Computing Heuristic
    Below is the thought process for computing the heuristic for a given state:

    1.  **Identify Goal Nuts:** Determine the set of nuts that are required to be in the `tightened` state according to the task's goals. This is done once during initialization.
    2.  **Identify Loose Goal Nuts:** In the current state, find which of the nuts identified in step 1 are still in the `loose` state. Let this set be `LooseGoalNuts`.
    3.  **Goal Check:** If `LooseGoalNuts` is empty, it means all goal nuts are tightened, so the goal is reached. The heuristic value is 0.
    4.  **Base Cost (Tighten Actions):** If there are loose goal nuts, a minimum of `num_loose_goal = |LooseGoalNuts|` `tighten_nut` actions are required. Add `num_loose_goal` to the total heuristic cost.
    5.  **Identify Man Agent and Location:** Find the name of the man agent. This is done by looking for an object in an `at` fact that is not a nut (involved in `loose`/`tightened` facts) or a spanner (involved in `usable`/`carrying` facts). Once the man is identified, find his current location from the `at` fact involving him.
    6.  **Count Carried Usable Spanners:** Determine how many usable spanners the man is currently carrying by checking `carrying` and `usable` facts in the state.
    7.  **Calculate Needed Pickups:** The man needs one usable spanner for each loose goal nut. Calculate how many additional usable spanners he needs to pick up from the ground: `needed_pickups = max(0, num_loose_goal - num_carried_usable)`.
    8.  **Cost for Pickups:** Add `needed_pickups` to the total heuristic cost, as each requires a `pickup_spanner` action.
    9.  **Identify Relevant Locations:** Determine the set of distinct locations the man needs to visit. This set includes:
        -   The location of each nut in `LooseGoalNuts`.
        -   The location of each usable spanner that is currently on the ground (not carried by the man), but *only if* `needed_pickups > 0`. Adding all usable ground spanner locations is a simple approximation of where pickups might occur.
    10. **Estimate Movement Cost:** Estimate the number of `walk` actions required for the man to visit all distinct locations in `RelevantLocations`. A simple, non-admissible estimate is used: If there are relevant locations, add 1 for the initial move towards the first relevant location (if the man isn't already at one), plus `|RelevantLocations| - 1` for subsequent moves between the remaining relevant locations. Add this `MovementCost` to the total heuristic.
    11. **Total Heuristic:** The final heuristic value is the sum of the base cost (tighten actions), pickup costs, and estimated movement cost.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal nuts.
        """
        self.goals = task.goals

        # Identify nuts that need to be tightened in the goal
        self.goal_nuts = set()
        for goal in self.goals:
            # Goal facts are typically positive literals like (tightened nut1)
            if match(goal, "tightened", "*"):
                _, nut_name = get_parts(goal)
                self.goal_nuts.add(nut_name)

        # Static facts like 'link' are not used in this version of the heuristic
        # to keep it simple and avoid shortest path calculations.
        # static_facts = task.static

    def __call__(self, node):
        """Compute an estimate of the minimal number of required actions."""
        state = node.state

        # 1. Identify loose goal nuts
        loose_goal_nuts = {
            nut for nut in self.goal_nuts if f'(loose {nut})' in state
        }

        # 2. If no loose goal nuts, goal is reached
        num_loose_goal = len(loose_goal_nuts)
        if num_loose_goal == 0:
            return 0

        # Base cost: one tighten action per loose goal nut
        total_cost = num_loose_goal

        # 3. Find the man agent name and location
        man_name = None
        # Identify all nuts and spanners first to exclude them
        all_nuts_in_state = {get_parts(fact)[1] for fact in state if match(fact, "loose", "*") or match(fact, "tightened", "*")}
        all_spanners_in_state = {get_parts(fact)[1] for fact in state if match(fact, "usable", "*") or match(fact, "carrying", "*", "*")}

        # The man is assumed to be the only locatable object that is not a nut or spanner
        man_location = None
        for fact in state:
            if match(fact, "at", "*", "*"):
                obj_name, loc = get_parts(fact)[1:3]
                if obj_name not in all_nuts_in_state and obj_name not in all_spanners_in_state:
                    man_name = obj_name
                    man_location = loc
                    break # Assuming only one man

        # If man_name wasn't found (unexpected in valid problems), return current cost
        if man_name is None:
             return total_cost # Cannot proceed with location/carrying checks

        # 4. Count carried usable spanners
        carried_usable_spanners = set()
        for fact in state:
             if match(fact, "carrying", man_name, "*"):
                 _, _, s_name = get_parts(fact)
                 if f'(usable {s_name})' in state:
                     carried_usable_spanners.add(s_name)

        num_carried_usable = len(carried_usable_spanners)

        # 5. Calculate needed pickups
        needed_pickups = max(0, num_loose_goal - num_carried_usable)
        total_cost += needed_pickups # Cost for pickup actions

        # 6. Build Relevant Locations
        relevant_locations = set()

        # Add locations of loose goal nuts
        for nut in loose_goal_nuts:
            for fact in state:
                if match(fact, "at", nut, "*"):
                    _, _, loc = get_parts(fact)
                    relevant_locations.add(loc)
                    break # Assuming nut is at only one location

        # Add locations of usable ground spanners if needed
        if needed_pickups > 0:
             # Find usable spanners on the ground and add their locations
             for fact in state:
                 if match(fact, "at", "*", "*"):
                     obj_name, loc = get_parts(fact)[1:3]
                     # Check if this object is a usable spanner on the ground
                     if f'(usable {obj_name})' in state and f'(carrying {man_name} {obj_name})' not in state:
                        relevant_locations.add(loc)

        # 7. Estimate Movement Cost
        movement_cost = 0
        if relevant_locations: # If there are locations the man needs to visit
            if man_location not in relevant_locations:
                # Man needs to move to the first relevant location
                movement_cost += 1
            # Man needs to move between the remaining relevant locations
            if len(relevant_locations) > 1:
                 movement_cost += len(relevant_locations) - 1

        total_cost += movement_cost

        return total_cost
