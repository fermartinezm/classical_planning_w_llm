import fnmatch
from heuristics.heuristic_base import Heuristic
from typing import Dict, Set, FrozenSet, Tuple, List, Optional

# Helper functions
def get_parts(fact: str) -> List[str]:
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    return fact[1:-1].split()

def match(fact: str, *args: str) -> bool:
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    # Ensure the number of parts in the fact matches the number of pattern arguments
    if len(parts) != len(args):
        return False
    # Check if each part matches the corresponding pattern argument using fnmatch
    return all(fnmatch.fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the PDDL domain 'childsnacks'.

    # Summary
    This heuristic estimates the number of actions required to serve all goal children.
    It calculates the sum of estimated actions needed for making sandwiches,
    putting them on trays, moving trays to the children's locations, and serving the children.
    The estimation tries to account for existing sandwiches and their locations,
    as well as gluten-free requirements for allergic children. It assumes resources like
    ingredients and trays are available when needed for the estimation, focusing on the
    minimum number of *types* of actions required.

    # Assumptions
    - The goal is always to have specific children served `(served childX)`.
    - Static predicates like `allergic_gluten`, `waiting`, `no_gluten_bread`, etc.,
      correctly define the fixed properties of the problem instance.
    - The heuristic does not need to be admissible.
    - For estimation purposes, it assumes sufficient ingredients and trays exist
      to perform necessary actions, even if not strictly true in the state (this simplifies
      computation and is acceptable for a non-admissible heuristic).
    - Object names follow conventions (e.g., trays start with 'tray') or are distinguishable
      by context when parsing 'at' predicates.

    # Heuristic Initialization
    - Stores the set of children that need to be served according to the goal.
    - Parses static facts to create mappings for:
        - Child allergies (`child -> is_allergic`).
        - Child waiting locations (`child -> place`).
    - These precomputed structures allow for efficient lookups during heuristic evaluation.

    # Step-By-Step Thinking for Computing Heuristic

    1.  **Identify Unserved Children:** Determine which children listed in the goal are not yet served in the current state. If all goal children are served, the heuristic value is 0.

    2.  **Base Cost (Serving):** Each unserved child requires one `serve` action. Initialize the heuristic `h` to the number of unserved children.

    3.  **Estimate Making Cost:**
        a. Count the total number of available sandwiches (both in the kitchen and on trays), distinguishing between gluten-free (GF) and regular.
        b. Count the number of unserved children requiring GF sandwiches and those requiring regular sandwiches.
        c. Match available sandwiches to needs, prioritizing GF sandwiches for GF needs. Use remaining GF sandwiches for regular needs if necessary.
        d. The number of children whose needs cannot be met by existing sandwiches estimates the number of `make_sandwich` actions required. Add this count to `h`.

    4.  **Estimate Putting Cost:**
        a. Count how many unserved children can potentially be satisfied by a sandwich that is *already on any tray* (regardless of the tray's current location). Match GF needs with GF sandwiches first.
        b. The difference between the total number of unserved children and the number satisfied by sandwiches already on trays estimates the number of `put_on_tray` actions needed (for sandwiches made or taken from the kitchen). Add this count to `h`.

    5.  **Estimate Moving Cost:**
        a. Identify, for each unserved child `c` waiting at place `p`, if there is a suitable sandwich (considering GF needs) on *any* tray currently located at `p`.
        b. Count the number of children who *cannot* be served by a sandwich on a tray already present at their location.
        c. This count estimates the minimum number of tray deliveries required (either moving a tray from the kitchen after making/putting, or moving a tray from another location). Add this count to `h`.

    6.  **Final Value:** The sum `h` represents the estimated total actions.
    """

    def __init__(self, task):
        """Initialize the heuristic by extracting goal conditions and static facts."""
        self.goals: FrozenSet[str] = task.goals
        self.static: FrozenSet[str] = task.static

        # Preprocess static facts for efficient lookup
        self.goal_children: Set[str] = {get_parts(fact)[1] for fact in self.goals if match(fact, "served", "*")}
        self.child_allergy: Dict[str, bool] = {}
        self.child_location: Dict[str, str] = {}
        # Note: We don't store gluten-free status of ingredients as the heuristic assumes
        # sufficient ingredients of the required type are available for making sandwiches.

        for fact in self.static:
            if match(fact, "allergic_gluten", "*"):
                self.child_allergy[get_parts(fact)[1]] = True
            elif match(fact, "not_allergic_gluten", "*"):
                child = get_parts(fact)[1]
                # Ensure not allergic is only set if no allergic fact exists
                if child not in self.child_allergy:
                    self.child_allergy[child] = False
            elif match(fact, "waiting", "*", "*"):
                child, place = get_parts(fact)[1:]
                self.child_location[child] = place
            # Ignore no_gluten_bread/content as per assumption

        # Ensure all goal children have an allergy entry (default to False if unspecified)
        for child in self.goal_children:
            if child not in self.child_allergy:
                self.child_allergy[child] = False


    def __call__(self, node) -> int:
        """Estimate the cost to reach the goal state from the given state node."""
        state: FrozenSet[str] = node.state
        all_facts: FrozenSet[str] = frozenset(state) # Use frozenset for efficient lookups ('in')

        # 1. Parse current state information
        served_children: Set[str] = {get_parts(fact)[1] for fact in state if match(fact, "served", "*")}

        # Identify unserved children based on goals
        unserved_children: Set[str] = self.goal_children - served_children
        if not unserved_children:
            return 0 # Goal state reached

        kitchen_sandwiches: Dict[str, bool] = {} # sandwich -> is_gluten_free
        sandwiches_on_tray: Dict[str, Tuple[str, bool]] = {} # sandwich -> (tray, is_gluten_free)
        tray_locations: Dict[str, str] = {} # tray -> place

        for fact in state:
            if match(fact, "at_kitchen_sandwich", "*"):
                s = get_parts(fact)[1]
                is_gf = f"(no_gluten_sandwich {s})" in all_facts
                kitchen_sandwiches[s] = is_gf
            elif match(fact, "ontray", "*", "*"):
                s, t = get_parts(fact)[1:]
                is_gf = f"(no_gluten_sandwich {s})" in all_facts
                sandwiches_on_tray[s] = (t, is_gf)
            elif match(fact, "at", "*", "*"):
                obj, loc = get_parts(fact)[1:]
                # Assume obj is a tray based on naming convention or context
                # A more robust check might involve querying object types if available
                if obj.startswith("tray"): # Simple assumption based on examples
                     tray_locations[obj] = loc

        # ----------------------------------------------------------------------
        # 2. Calculate heuristic value h
        # ----------------------------------------------------------------------
        num_serves = len(unserved_children)
        h = num_serves # Base cost: 1 serve action per child

        # --- Estimate num_makes_needed ---
        avail_gf_sandwiches_kitchen = {s for s, is_gf in kitchen_sandwiches.items() if is_gf}
        avail_reg_sandwiches_kitchen = {s for s, is_gf in kitchen_sandwiches.items() if not is_gf}
        avail_gf_sandwiches_tray = {s for s, (_, is_gf) in sandwiches_on_tray.items() if is_gf}
        avail_reg_sandwiches_tray = {s for s, (_, is_gf) in sandwiches_on_tray.items() if not is_gf}

        avail_gf_total = len(avail_gf_sandwiches_kitchen | avail_gf_sandwiches_tray)
        avail_reg_total = len(avail_reg_sandwiches_kitchen | avail_reg_sandwiches_tray)

        needed_gf = 0
        needed_regular = 0
        for c in unserved_children:
            if self.child_allergy.get(c, False):
                needed_gf += 1
            else:
                needed_regular += 1

        # Match GF needs first with available GF sandwiches
        match_gf_needs = min(needed_gf, avail_gf_total)
        needed_gf -= match_gf_needs
        remaining_gf_available = avail_gf_total - match_gf_needs

        # Match regular needs with available regular sandwiches
        match_reg_needs_regular = min(needed_regular, avail_reg_total)
        needed_regular -= match_reg_needs_regular

        # Match remaining regular needs with remaining available GF sandwiches
        match_reg_needs_gf = min(needed_regular, remaining_gf_available)
        needed_regular -= match_reg_needs_gf

        sandwiches_to_make = needed_gf + needed_regular # Remaining unmet needs require making
        h += sandwiches_to_make

        # --- Estimate num_puts_needed ---
        # Count children whose needs can be met by a sandwich currently on *any* tray.
        unserved_children_copy = list(unserved_children)
        # Create mutable lists of available tray sandwiches
        gf_tray_sandwiches = [s for s, (_, is_gf) in sandwiches_on_tray.items() if is_gf]
        reg_tray_sandwiches = [s for s, (_, is_gf) in sandwiches_on_tray.items() if not is_gf]
        served_by_existing_tray_sandwich = 0

        # Try satisfy GF needs first with GF tray sandwiches
        indices_to_remove_child_gf = []
        for i, c in enumerate(unserved_children_copy):
            if self.child_allergy.get(c, False):
                if gf_tray_sandwiches:
                    served_by_existing_tray_sandwich += 1
                    indices_to_remove_child_gf.append(i)
                    gf_tray_sandwiches.pop() # Consume one GF tray sandwich

        # Remove matched children (iterate backwards)
        for i in sorted(indices_to_remove_child_gf, reverse=True):
            del unserved_children_copy[i]

        # Try satisfy remaining needs (regular + GF needs not met above) with any remaining tray sandwich
        indices_to_remove_child_rem = []
        available_tray_sandwiches = gf_tray_sandwiches + reg_tray_sandwiches # Pool remaining
        for i, c in enumerate(unserved_children_copy):
             if available_tray_sandwiches:
                 # Check if the available sandwich is suitable (GF child needs GF sandwich)
                 sandwich_to_use = None
                 sandwich_idx_to_use = -1

                 # Prioritize using remaining regular sandwiches for non-allergic children
                 if not self.child_allergy.get(c, False):
                     for k, s_name in enumerate(available_tray_sandwiches):
                         # Find the type of this sandwich (check if it was originally regular)
                         _, s_is_gf = sandwiches_on_tray[s_name]
                         if not s_is_gf:
                             sandwich_to_use = s_name
                             sandwich_idx_to_use = k
                             break
                 # If no suitable regular sandwich found, or if child is allergic, use any available (must be GF if child is allergic)
                 if sandwich_to_use is None:
                     for k, s_name in enumerate(available_tray_sandwiches):
                         _, s_is_gf = sandwiches_on_tray[s_name]
                         if self.child_allergy.get(c, False): # Allergic child
                             if s_is_gf: # Needs GF sandwich
                                 sandwich_to_use = s_name
                                 sandwich_idx_to_use = k
                                 break
                         else: # Non-allergic child, any sandwich works
                             sandwich_to_use = s_name
                             sandwich_idx_to_use = k
                             break

                 if sandwich_to_use is not None:
                     served_by_existing_tray_sandwich += 1
                     indices_to_remove_child_rem.append(i)
                     del available_tray_sandwiches[sandwich_idx_to_use] # Consume the sandwich

        # Remove matched children
        for i in sorted(indices_to_remove_child_rem, reverse=True):
            del unserved_children_copy[i]

        num_puts_needed = num_serves - served_by_existing_tray_sandwich
        h += num_puts_needed

        # --- Estimate num_moves_needed ---
        # Count children who cannot be served by a suitable sandwich on a tray *already at their location*.
        sandwiches_on_tray_at_loc: Dict[str, List[Tuple[str, bool]]] = {} # Map loc -> list of (sandwich, is_gf)
        for s, (t, is_gf) in sandwiches_on_tray.items():
            loc = tray_locations.get(t)
            if loc:
                if loc not in sandwiches_on_tray_at_loc: sandwiches_on_tray_at_loc[loc] = []
                sandwiches_on_tray_at_loc[loc].append((s, is_gf))

        unserved_children_copy = list(unserved_children)
        # Create a mutable copy of sandwiches available at each location
        temp_sandwiches_at_loc = {loc: list(sands) for loc, sands in sandwiches_on_tray_at_loc.items()}
        served_without_move = 0

        # Try satisfy GF needs first with GF tray sandwiches *at the location*
        indices_to_remove_child_gf_loc = []
        for i, c in enumerate(unserved_children_copy):
            p = self.child_location.get(c)
            if not p: continue # Should have a location
            if self.child_allergy.get(c, False):
                if p in temp_sandwiches_at_loc:
                    found_gf_at_loc = False
                    # Iterate through sandwiches at location p to find a GF one
                    for j in range(len(temp_sandwiches_at_loc[p]) - 1, -1, -1):
                        s_loc, is_gf_loc = temp_sandwiches_at_loc[p][j]
                        if is_gf_loc:
                            served_without_move += 1
                            indices_to_remove_child_gf_loc.append(i)
                            del temp_sandwiches_at_loc[p][j] # Consume sandwich
                            found_gf_at_loc = True
                            break # Found one for this child
                    if found_gf_at_loc: continue # Move to next child

        # Remove matched children
        for i in sorted(indices_to_remove_child_gf_loc, reverse=True):
            del unserved_children_copy[i]

        # Try satisfy remaining needs with any suitable sandwich *at the location*
        indices_to_remove_child_rem_loc = []
        for i, c in enumerate(unserved_children_copy):
            p = self.child_location.get(c)
            if not p: continue
            if p in temp_sandwiches_at_loc:
                 found_suitable_at_loc = False
                 # Iterate through remaining sandwiches at location p
                 for j in range(len(temp_sandwiches_at_loc[p]) - 1, -1, -1):
                     s_loc, is_gf_loc = temp_sandwiches_at_loc[p][j]
                     # Check suitability
                     is_suitable = False
                     if self.child_allergy.get(c, False):
                         is_suitable = is_gf_loc # Allergic child needs GF
                     else:
                         is_suitable = True # Non-allergic child can have any

                     if is_suitable:
                         served_without_move += 1
                         indices_to_remove_child_rem_loc.append(i)
                         del temp_sandwiches_at_loc[p][j] # Consume sandwich
                         found_suitable_at_loc = True
                         break # Found one for this child
                 if found_suitable_at_loc: continue # Move to next child

        # Remove matched children
        for i in sorted(indices_to_remove_child_rem_loc, reverse=True):
            del unserved_children_copy[i]

        num_moves_needed = num_serves - served_without_move
        h += num_moves_needed

        return h

