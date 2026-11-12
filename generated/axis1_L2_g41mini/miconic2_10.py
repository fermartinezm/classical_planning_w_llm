from heuristics.heuristic_base import Heuristic


def get_parts(fact):
    """Extract the predicate and arguments from a PDDL fact string by stripping parentheses and splitting."""
    return fact[1:-1].split()


class NewDomainDependentHeuristic(Heuristic):
    """
    Domain-dependent heuristic for the secret_domain domain (unkhrp2l).

    Summary
    -------
    This heuristic estimates the number of actions required to achieve the goal conditions focusing
    mainly on transforming all relevant jxhpkqr5 objects from the "eacj8ftl" to "guby8wxu" predicate,
    as the goal is expressed as (guby8wxu ?x). It uses the static relations and current state to estimate
    how many actions remain to transform all needed jxhpkqr5 objects.

    Assumptions
    -----------
    - The goal is to have certain jxhpkqr5 objects hold predicate guby8wxu.
    - Objects of type jxhpkqr5 currently in eacj8ftl predicate must be transformed to guby8wxu by an action that needs a related ql6b2h31 object.
    - ql6b2h31 objects have connectivity via predicate pvhp8m0p, modeling possible moves of a "flag" (v0yy74n7) between them.
    - An object of type jxhpkqr5 transitions from pxch7pi8 to eacj8ftl and then to guby8wxu, making the pxch7pi8 and pahbmn6t predicates relevant for intermediate states.
    - There is exactly one "flag" (v0yy74n7) fact true at a time, representing a "token" or position moving between ql6b2h31 objects.
    - The heuristic does not require admissibility but should be informative to guide greedy best-first search efficiently.

    Heuristic Initialization
    ------------------------
    - Extract goal jxhpkqr5 objects that must have the (guby8wxu ?obj) fact true.
    - Parse static facts to build dictionaries:
      * For each jxhpkqr5 object, map its associated pxch7pi8 (linked ql6b2h31).
      * For each jxhpkqr5 object, map its associated pahbmn6t (linked ql6b2h31).
      * For ql6b2h31 objects, build adjacency from pvhp8m0p to allow shortest-path computations.
    - These data structures enable estimation of travel steps and required state transitions.

    Step-By-Step Thinking for Computing Heuristic
    ---------------------------------------------
    1. Identify all jxhpkqr5 objects in the goal (guby8wxu) to compute remaining needed transformations.
    2. For each goal object not yet guby8wxu, estimate cost to:
       a. If it is currently eacj8ftl on some ql6b2h31, one action (jg64qouf) is needed to make it guby8wxu, provided the flag is on that ql6b2h31.
       b. Otherwise, if it's pxch7pi8 on some ql6b2h31, then first the object must be switched (makpz2tc) to eacj8ftl, which requires the object linked to its pxch7pi8 ql6b2h31 matching the flag location.
          Then from eacj8ftl, as in step (a), transform it into guby8wxu (jg64qouf).
       c. To reliably perform these actions, the flag (v0yy74n7) must be moved to the required ql6b2h31 objects.
    3. Estimate flag movements required:
       - Given the flag's current ql6b2h31, compute shortest path lengths to the pxch7pi8 and/or pahbmn6t ql6b2h31 locations needed by the object.
    4. Sum minimum estimated costs for all goal objects not yet guby8wxu:
       - Flag move actions (each jt9wnwgs or v90zoaro moves flag between adjacent ql6b2h31).
       - The required object's predicate changes (makpz2tc or jg64qouf).
    5. As actions always involve single step movements or one predicate change, add costs straightforwardly assuming independence.
    6. Return 0 if all goals reached; otherwise, the summed estimated action counts guiding the search toward moving flag and changing predicates accordingly.
    """

    def __init__(self, task):
        """Extract goal objects, static relations, and prepare adjacency graph of ql6b2h31 objects."""
        self.goals = task.goals
        self.static = task.static

        # Extract jxhpkqr5 objects required to reach goal (those in (guby8wxu ?obj))
        self.goal_objs = set()
        for fact in self.goals:
            parts = get_parts(fact)
            if parts[0] == "guby8wxu":
                self.goal_objs.add(parts[1])

        # Dictionaries:
        # pxch7pi8: object_jxhpkqr5 -> ql6b2h31 object
        # pahbmn6t: object_jxhpkqr5 -> ql6b2h31 object
        # adjacency: ql6b2h31 -> set of adjacent ql6b2h31 (from pvhp8m0p edges)
        self.pxch7pi8_map = {}
        self.pahbmn6t_map = {}
        self.adjacency = {}

        # Collect all ql6b2h31 objects seen
        self.all_ql6b2h31 = set()
        for fact in self.static:
            parts = get_parts(fact)
            if parts[0] == "pxch7pi8":
                obj_jxh = parts[1]
                obj_ql = parts[2]
                self.pxch7pi8_map[obj_jxh] = obj_ql
                self.all_ql6b2h31.add(obj_ql)
            elif parts[0] == "pahbmn6t":
                obj_jxh = parts[1]
                obj_ql = parts[2]
                self.pahbmn6t_map[obj_jxh] = obj_ql
                self.all_ql6b2h31.add(obj_ql)
            elif parts[0] == "pvhp8m0p":
                frm = parts[1]
                to = parts[2]
                self.all_ql6b2h31.add(frm)
                self.all_ql6b2h31.add(to)
                if frm not in self.adjacency:
                    self.adjacency[frm] = set()
                self.adjacency[frm].add(to)
        # Ensure all ql6b2h31 keys present in adjacency (for neighbors)
        for loc in self.all_ql6b2h31:
            if loc not in self.adjacency:
                self.adjacency[loc] = set()

    def _shortest_flag_moves(self, start, target):
        """
        Compute shortest number of flag moves between two ql6b2h31 nodes using BFS.
        Returns the length of shortest path or a large number if unreachable.
        """
        if start == target:
            return 0
        visited = set()
        queue = [(start, 0)]
        while queue:
            current, dist = queue.pop(0)
            if current == target:
                return dist
            if current in visited:
                continue
            visited.add(current)
            neighbors = self.adjacency.get(current, set())
            queue.extend((nbr, dist + 1) for nbr in neighbors if nbr not in visited)
            # Also consider edges reversed since pvhp8m0p is directional, but two actions exist
            # to move flag backward (jt9wnwgs and v90zoaro), so consider reverse edges too.
            for loc in self.all_ql6b2h31:
                if current in self.adjacency.get(loc, set()) and loc not in visited:
                    queue.append((loc, dist + 1))
        # Unreachable, return large cost
        return 100000

    def __call__(self, node):
        state = node.state

        # Quick goal check
        if all(f"(guby8wxu {obj})" in state for obj in self.goal_objs):
            return 0

        # Find current flag location: there is exactly one (v0yy74n7 ?ql6b2h31)
        flag_pos = None
        for fact in state:
            parts = get_parts(fact)
            if parts[0] == "v0yy74n7":
                flag_pos = parts[1]
                break
        if flag_pos is None:
            # No flag found (should not happen); return large heuristic
            return 100000

        # Gather predicate info per jxhpkqr5 object in the goal set:
        # Track if in guby8wxu, eacj8ftl, pxch7pi8 (we do not care otherwise)
        objects_status = {}
        for obj in self.goal_objs:
            objects_status[obj] = {
                "guby8wxu": False,
                "eacj8ftl": False,
                "pxch7pi8": False,
            }

        # Scan state once, record status for goal objects
        for fact in state:
            parts = get_parts(fact)
            pred = parts[0]
            if pred in objects_status and parts[1] in self.goal_objs:
                obj = parts[1]
                if pred in objects_status[obj]:
                    objects_status[obj][pred] = True

        total_cost = 0

        # Action costs from domain:
        # Move flag costs: 1 action per move (jt9wnwgs or v90zoaro)
        flag_move_cost = 1
        # State change actions for jxhpkqr5 objects:
        # makpz2tc: pxch7pi8(obj, ql6b2h31) -> eacj8ftl(obj)  [requires flag at pxch7pi8 ql6b2h31]
        # jg64qouf: eacj8ftl(obj) -> guby8wxu(obj)           [requires flag at pahbmn6t ql6b2h31]
        # Each of these actions cost 1.

        for obj in self.goal_objs:
            if objects_status[obj]["guby8wxu"]:
                # Already done for this object
                continue

            # Object still not guby8wxu, estimate cost to reach goal for this object.

            # Case 1: object currently eacj8ftl -> need 1 jg64qouf action at pahbmn6t location of obj
            if objects_status[obj]["eacj8ftl"]:
                # Flag must be at pahbmn6t[obj] for jg64qouf
                target_loc = self.pahbmn6t_map.get(obj, None)
                if target_loc is None:
                    # No known location for action, assign large cost (should not happen)
                    total_cost += 100000
                    continue
                move_steps = self._shortest_flag_moves(flag_pos, target_loc)
                # 1 action for predicate change + moves
                total_cost += move_steps * flag_move_cost + 1

                # After doing jg64qouf here, object reaches goal - so flag_pos virtually moves to target_loc
                flag_pos = target_loc

            # Case 2: object currently pxch7pi8 -> must first do makpz2tc at pxch7pi8 location,
            # then jg64qouf at pahbmn6t location.
            elif objects_status[obj]["pxch7pi8"]:
                loc_pxch = self.pxch7pi8_map.get(obj, None)
                loc_pahb = self.pahbmn6t_map.get(obj, None)
                if loc_pxch is None or loc_pahb is None:
                    # Unknown relevant locations; large cost
                    total_cost += 100000
                    continue

                # Move flag to pxch7pi8 location for makpz2tc first
                move_to_pxch = self._shortest_flag_moves(flag_pos, loc_pxch)

                # After makpz2tc, object becomes eacj8ftl and flag stays at loc_pxch
                # Then move flag to pahbmn6t location for jg64qouf
                move_to_pahb = self._shortest_flag_moves(loc_pxch, loc_pahb)

                # Actions:
                # makpz2tc 1 action
                # jg64qouf 1 action
                total_cost += (move_to_pxch + move_to_pahb) * flag_move_cost + 2

                # Flag ends at pahbmn6t location after jg64qouf
                flag_pos = loc_pahb

            else:
                # Object not in eacj8ftl or pxch7pi8 predicates (probably not prepared)
                # We must assume it must be first made pxch7pi8 or eacj8ftl - no such info easily available,
                # assign conservative large cost to prioritize exploring states fixing this.
                total_cost += 20

        return total_cost
