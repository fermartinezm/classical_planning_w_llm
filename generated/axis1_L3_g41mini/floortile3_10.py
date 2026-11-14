from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()


def match(fact, *args):
    """
    Check if a PDDL fact matches a pattern.

    - `fact`: Complete fact string, e.g. "(tightened nut1 locatable1)".
    - `args`: Pattern elements, with possible wildcards "*".
    - Returns True if fact matches pattern; False otherwise.
    """
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class NewDomainDependentHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the 'secret_domain' (spanner) domain.

    Summary:
    --------
    This heuristic estimates the minimal number of actions needed to achieve all
    goals of the form (tightened-2 locatable location) from the current state.
    It focuses on the main tasks: moving tightened nuts across locatables,
    ensuring nuts are tightened at correct locations, and updating loose nuts'
    locations.

    Assumptions:
    ------------
    - Goal facts are always of the form (tightened-2 locatable location).
    - Only locatables that appear in 'link' predicate can be moved or swapped.
    - Nuts must be tightened onto specific locatables and finally placed at
      specific locations, represented by 'tightened-2' facts.
    - Movement of "tightened" relations corresponds roughly to relocating nuts.
    - Loose nuts need to be eventually tightened relative to locatables.
    - Each movement/tightening action counts approximately as one step.
    - Some preconditions require loose-2 facts and locations; heuristic assumes
      moving a nut from one location to another loose location costs one action.
    - The heuristic is not admissible but should guide greedy search efficiently.

    Heuristic Initialization:
    -------------------------
    - Extract all goals of form (tightened-2 locatable location).
    - Extract static info:
      * Maps of looseness relations (loose loc1 loc2) → adjacent locatables.
      * Which locatables are linked (link predicate).
      * At predicates indicating adjacency between locatables.
      * Usable predicates indicating possible further movements.
    - Extract all locations (at-2 facts).
    - Store these relations in dictionaries/sets for quick lookup.
    - Precompute item sets for quick reference during heuristic evaluation.

    Step-By-Step Thinking for Computing Heuristic:
    ----------------------------------------------
    1. If state satisfies goals (all required (tightened-2 locatable location)), heuristic = 0.

    2. For each goal (tightened-2 locatable location), check current state:
       a. Identify where the given locatable currently has the tightened relation.
          - From state facts (tightened-2 locatable location') or (tightened nut locatable').
          - If current location matches goal location, no cost for this goal.
       b. If not at goal location, there is at least one movement needed.
          Moving a tightened association from locatable loc1 to locatable loc2
          requires roughly one "walk-2", "tighten_nut-2", or similar action.

    3. For each nut, if loose-2 (nut location) does not match its expected final location,
       it needs at least one tighten_nut action.

    4. Sum costs for each mismatched goal fact and loose-nut placements.

    5. Add extra cost heuristically for moving nuts through the "link" chain
       when tightness must be transferred between locatables.

    This heuristic counts as 1 the cost to relocate each nut/locatable relationship
    towards its goal placement, ignoring complex intermediate states.
    """

    def __init__(self, task):
        self.goals = task.goals
        self.static = task.static

        # Extract goal map: locatable -> goal location for tightened-2 facts
        self.goal_tightened2 = {}
        for goal in self.goals:
            parts = get_parts(goal)
            if parts[0] == "tightened-2":
                locatable, location = parts[1], parts[2]
                self.goal_tightened2[locatable] = location

        # Extract all loose relations: loose loc1 loc2 (adjacent locatables)
        self.loose_adjacents = set()
        for fact in self.static:
            if match(fact, "loose", "*", "*"):
                parts = get_parts(fact)
                self.loose_adjacents.add((parts[1], parts[2]))

        # Extract link locatables (which locatables can move/be linked)
        self.linked_locatables = set()
        for fact in self.static:
            if match(fact, "link", "*"):
                parts = get_parts(fact)
                self.linked_locatables.add(parts[1])

        # Extract usable relations (usable loc1 loc2)
        self.usable_links = set()
        for fact in self.static:
            if match(fact, "usable", "*", "*"):
                parts = get_parts(fact)
                self.usable_links.add((parts[1], parts[2]))

        # Extract all locations (at-2 facts)
        self.locations = set()
        for fact in self.static:
            if match(fact, "at-2", "*"):
                parts = get_parts(fact)
                self.locations.add(parts[1])

    def __call__(self, node):
        state = node.state

        # Quick check: if goals satisfied, heuristic=0
        goal_facts = {f for f in state if f.startswith("(tightened-2")}
        if self.goals <= goal_facts:
            return 0

        # Extract current tightened-2 facts: locatable -> location
        curr_tightened2 = {}
        for fact in state:
            if match(fact, "tightened-2", "*", "*"):
                parts = get_parts(fact)
                curr_tightened2[parts[1]] = parts[2]

        # Extract tightened facts: nut -> locatable
        tightened_nuts = {}
        for fact in state:
            if match(fact, "tightened", "*", "*"):
                parts = get_parts(fact)
                tightened_nuts[parts[1]] = parts[2]

        # Extract loose-2 facts: nut -> location
        loose_nuts_locations = {}
        for fact in state:
            if match(fact, "loose-2", "*", "*"):
                parts = get_parts(fact)
                loose_nuts_locations[parts[1]] = parts[2]

        # Count of estimated needed steps
        cost = 0

        # 1) Cost to fix each tightened-2 locatable placement to goal location
        for locatable, goal_loc in self.goal_tightened2.items():
            curr_loc = curr_tightened2.get(locatable)
            if curr_loc != goal_loc:
                # Moving tightened-2 from curr_loc to goal_loc costs at least 1
                cost += 1

        # 2) Cost to tighten loose nuts at their goal location if applicable:
        # - For each nut that is loose-2 at location l, estimate if it must be tightened at goal location
        # We estimate this cost only if loose nut location differs from location of a linked tightened locatable or goal location.
        for nut, loose_loc in loose_nuts_locations.items():
            # Find current tightened locatable where this nut is "tightened"
            tightened_locatable = tightened_nuts.get(nut)

            # If loose-2 location differs from current location of nut or goal, tightening needed -> +1 cost
            # Approximate: Each loose nut must be tightened at a location eventually
            # If loose location matches any of tightened locatable's location, no extra cost
            if tightened_locatable is None:
                # Nut is loose only, must be tightened at some location => count cost
                cost += 1
            else:
                curr_tightened2_loc = curr_tightened2.get(tightened_locatable)
                goal_loc_for_tightened = self.goal_tightened2.get(tightened_locatable)
                if (loose_loc != curr_tightened2_loc) or (goal_loc_for_tightened and loose_loc != goal_loc_for_tightened):
                    cost += 1

        return cost
