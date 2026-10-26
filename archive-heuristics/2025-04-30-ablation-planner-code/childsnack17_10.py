from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract the components of a PDDL fact by removing parentheses and splitting the string."""
    # Ensure the fact is a string and has parentheses
    if not isinstance(fact, str) or not fact.startswith('(') or not fact.endswith(')'):
        return []
    return fact[1:-1].split()

def match(fact, *args):
    """
    Check if a PDDL fact matches a given pattern.

    - `fact`: The complete fact as a string, e.g., "(at tray1 kitchen)".
    - `args`: The expected pattern (wildcards `*` allowed).
    - Returns `True` if the fact matches the pattern, else `False`.
    """
    parts = get_parts(fact)
    if len(parts) != len(args):
        return False
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))


class childsnackHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the childsnacks domain.

    # Summary
    This heuristic estimates the minimum number of actions required to serve all
    waiting children. It counts the necessary 'make_sandwich', 'put_on_tray',
    'move_tray', and 'serve_sandwich' actions based on the current state and
    the children's needs and locations.

    # Assumptions
    - The heuristic ignores resource constraints like the exact number of
      available bread/content portions or 'notexist' sandwich slots beyond
      what's needed to satisfy the count of required sandwiches.
    - It assumes enough trays are available in the kitchen for 'put_on_tray'
      actions if needed.
    - It simplifies tray movements by counting the number of distinct locations
      (excluding the kitchen) where unserved children are waiting but no tray
      is currently present, assuming one move action is sufficient to bring
      a tray to each such location.
    - A gluten-free sandwich can serve any child, while a regular sandwich
      can only serve a child who is not allergic to gluten.

    # Heuristic Initialization
    - Extracts the set of all children that need to be served from the goal
      conditions.
    - Extracts the allergy status (allergic_gluten) for each child from the
      static facts.

    # Step-By-Step Thinking for Computing Heuristic
    The heuristic value is the sum of estimated costs for four main types of
    actions required to serve all unserved children: making sandwiches, putting
    sandwiches on trays, moving trays to children's locations, and serving
    the children.

    1.  **Identify Unserved Children and Their Needs:**
        - Determine which children from the goal list have not yet been served
          in the current state.
        - For each unserved child, find their current waiting location (from
          the state) and their allergy status (from initialization).
        - Count the total number of unserved children needing a gluten-free
          sandwich (allergic children) and those needing any sandwich
          (non-allergic children).

    2.  **Count Available Sandwiches:**
        - Count how many gluten-free and regular sandwiches are currently
          available, either in the kitchen or already on trays.
        - Count how many gluten-free and regular sandwiches are already on trays.

    3.  **Estimate 'Make Sandwich' Cost:**
        - Calculate the number of gluten-free sandwiches that still need to be
          made: `max(0, needed_gf_children - avail_gf_sandwiches)`.
        - Calculate the number of regular sandwiches that still need to be made.
          Non-allergic children can use any sandwich. We prioritize using
          available GF sandwiches for allergic children first. The remaining
          GF sandwiches, plus available regular sandwiches, can serve
          non-allergic children.
          `make_reg = max(0, needed_reg_children - max(0, avail_gf_sandwiches - needed_gf_children) - avail_reg_sandwiches)`.
        - The cost is the sum of `make_gf` and `make_reg`.

    4.  **Estimate 'Put on Tray' Cost:**
        - Calculate the number of gluten-free sandwiches that need to be put
          on trays: `max(0, needed_gf_children - ontray_gf_sandwiches)`.
        - Calculate the number of regular sandwiches that need to be put on
          trays. Similar to making, non-allergic children's needs can be met
          by leftover GF sandwiches already on trays, plus regular sandwiches
          already on trays.
          `put_reg = max(0, needed_reg_children - max(0, ontray_gf_sandwiches - needed_gf_children) - ontray_reg_sandwiches)`.
        - The cost is the sum of `put_gf` and `put_reg`. This assumes enough
          trays are available in the kitchen.

    5.  **Estimate 'Move Tray' Cost:**
        - Identify all distinct locations (excluding the kitchen) where
          unserved children are waiting.
        - Identify all locations where trays are currently present.
        - Count the number of waiting locations (excluding kitchen) that do
          not currently have a tray. Each such location requires at least one
          tray movement action to bring a tray there.
        - The cost is the count of these locations.

    6.  **Estimate 'Serve Sandwich' Cost:**
        - Each unserved child requires one 'serve' action.
        - The cost is the total number of unserved children.

    7.  **Total Heuristic:**
        - The total heuristic value is the sum of the estimated costs for
          make, put, move, and serve actions.
    """

    def __init__(self, task):
        """
        Initialize the heuristic by extracting goal children and child allergy status.
        """
        self.goals = task.goals
        self.static_facts = task.static

        # Identify all children that need to be served from the goal
        self.children_to_serve = set()
        for goal in self.goals:
            parts = get_parts(goal)
            if parts and parts[0] == 'served':
                if len(parts) > 1:
                    self.children_to_serve.add(parts[1])

        # Map child name to allergy status from static facts
        self.child_allergy = {}
        for fact in self.static_facts:
            parts = get_parts(fact)
            if parts and parts[0] in ['allergic_gluten', 'not_allergic_gluten']:
                if len(parts) > 1:
                    child = parts[1]
                    self.child_allergy[child] = (parts[0] == 'allergic_gluten')

    def __call__(self, node):
        """
        Compute an estimate of the minimal number of required actions to serve
        all children.
        """
        state = node.state

        # 1. Identify Unserved Children and Their Needs
        served_children = {get_parts(fact)[1] for fact in state if match(fact, 'served', '*')}
        
        # Filter children_to_serve to only include those present in the state's waiting facts
        # and not yet served. This handles cases where goal includes children not in init/state.
        waiting_children_info = {} # child -> place
        for fact in state:
            parts = get_parts(fact)
            if parts and parts[0] == 'waiting' and len(parts) > 2:
                 child, place = parts[1], parts[2]
                 if child in self.children_to_serve and child not in served_children:
                     waiting_children_info[child] = place

        unserved_children = set(waiting_children_info.keys())

        if not unserved_children:
            return 0 # Goal is reached

        unserved_needs = {
            child: {
                'place': waiting_children_info[child],
                'allergic': self.child_allergy.get(child, False) # Default to not allergic if status not specified
            }
            for child in unserved_children
        }

        needed_gf_children = sum(1 for need in unserved_needs.values() if need['allergic'])
        needed_reg_children = sum(1 for need in unserved_needs.values() if not need['allergic'])

        # 2. Count Available Sandwiches
        kitchen_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, 'at_kitchen_sandwich', '*')}
        ontray_sandwiches_map = {get_parts(fact)[1]: get_parts(fact)[2] for fact in state if match(fact, 'ontray', '*', '*')} # sandwich -> tray
        gluten_free_sandwiches = {get_parts(fact)[1] for fact in state if match(fact, 'no_gluten_sandwich', '*')}

        avail_gf_sandwiches = 0
        avail_reg_sandwiches = 0
        ontray_gf_sandwiches = 0
        ontray_reg_sandwiches = 0

        # Sandwiches in kitchen
        for s in kitchen_sandwiches:
            if s in gluten_free_sandwiches:
                avail_gf_sandwiches += 1
            else:
                avail_reg_sandwiches += 1

        # Sandwiches on trays
        for s in ontray_sandwiches_map.keys():
             if s in gluten_free_sandwiches:
                 avail_gf_sandwiches += 1
                 ontray_gf_sandwiches += 1
             else:
                 avail_reg_sandwiches += 1
                 ontray_reg_sandwiches += 1

        # 3. Estimate 'Make Sandwich' Cost
        make_gf = max(0, needed_gf_children - avail_gf_sandwiches)
        # Regular children can use any sandwich. Prioritize using available GF for allergic.
        # Available GF for regular children = max(0, avail_gf_sandwiches - needed_gf_children)
        # Available Reg for regular children = avail_reg_sandwiches
        # Total available for regular children = max(0, avail_gf_sandwiches - needed_gf_children) + avail_reg_sandwiches
        make_reg = max(0, needed_reg_children - (max(0, avail_gf_sandwiches - needed_gf_children) + avail_reg_sandwiches))
        cost_make = make_gf + make_reg

        # 4. Estimate 'Put on Tray' Cost
        # Need needed_gf_children GF sandwiches on trays. Have ontray_gf_sandwiches.
        put_gf = max(0, needed_gf_children - ontray_gf_sandwiches)
        # Need needed_reg_children Reg sandwiches on trays. Available on trays for reg children
        # = max(0, ontray_gf_sandwiches - needed_gf_children) + ontray_reg_sandwiches
        put_reg = max(0, needed_reg_children - (max(0, ontray_gf_sandwiches - needed_gf_children) + ontray_reg_sandwiches))
        cost_put = put_gf + put_reg

        # 5. Estimate 'Move Tray' Cost
        waiting_locations = {need['place'] for need in unserved_needs.values() if need['place'] is not None}
        
        tray_locations = {} # tray -> place
        for fact in state:
             parts = get_parts(fact)
             if parts and parts[0] == 'at' and len(parts) > 2 and parts[1].startswith('tray'):
                  tray_locations[parts[1]] = parts[2]

        tray_at_location = set(tray_locations.values()) # Set of locations with trays

        locations_needing_tray = {
            p for p in waiting_locations
            if p != 'kitchen' and p not in tray_at_location
        }
        cost_move = len(locations_needing_tray)

        # 6. Estimate 'Serve Sandwich' Cost
        cost_serve = len(unserved_children)

        # 7. Total Heuristic
        total_cost = cost_make + cost_put + cost_move + cost_serve

        return total_cost

