from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

class secret_domainHeuristic(Heuristic):
    """
    A domain-dependent heuristic for the secret_domain (r9oudocu) domain.

    # Summary
    This heuristic estimates the number of actions required to achieve the goal by considering:
    1. The number of missing `xk87au5b` goals multiplied by 2 (each requires converting to state B and then applying action2)
    2. The number of missing `dhjk1eyy` goals
    3. The number of missing `x272hpoe` relations
    4. An extra cost for each relation where the source object isn't in state B (requires conversion)
    5. An additional cost if the flag isn't set and there are state A objects needing conversion for `xk87au5b`

    # Assumptions
    - Objects can be in three states:
        State A: Has both `dhjk1eyy` and `xk87au5b` (not in state B)
        State B: Has `q71n8mtz` (and not State A)
        Other: Mixed states (e.g., only one predicate) are possible but handled conservatively
    - Converting an object from state A to state B requires 1 action (action1)
    - Setting `xk87au5b` requires 1 action (action2) if the object is already in state B, or 2 actions (convert then set) otherwise
    - Creating a relation requires 1 action (action3) if the source object is in state B, or 2 actions (convert then create) otherwise
    - The flag state may require an extra action when missing for certain conversions

    # Heuristic Initialization
    - Extracts goal conditions for `dhjk1eyy`, `xk87au5b`, and `x272hpoe` from the task
    - Initializes data structures to track goal objects and relations

    # Step-By-Step Thinking for Computing Heuristic
    1. Parse the current state to determine:
        - Flag status (`hxtpdpff`)
        - Objects and their states (`dhjk1eyy`, `xk87au5b`, `q71n8mtz`)
        - Existing relations (`x272hpoe`)
    2. Identify missing goals:
        - `xk87au5b`: Objects that should have it but don't
        - `dhjk1eyy`: Objects that should have it but don't
        - `x272hpoe`: Relations that should exist but don't
    3. For each missing relation, check if the source object is in state B (has `q71n8mtz`). If not, add an extra cost for conversion.
    4. Identify state A objects (have both `dhjk1eyy` and `xk87au5b`) that are missing `xk87au5b` (need conversion to state B)
    5. Compute base costs:
        - 2 per missing `xk87au5b` (convert to state B + apply action2)
        - 1 per missing `dhjk1eyy`
        - 1 per missing relation
        - 1 per extra relation conversion (source not in state B)
    6. Add 1 if flag isn't set and there are state A objects needing conversion for `xk87au5b`
    """

    def __init__(self, task):
        """Initialize goal conditions from the task."""
        self.goals = task.goals
        self.static = task.static
        
        # Extract goal conditions
        self.goal_dhjk1eyy = set()
        self.goal_xk87au5b = set()
        self.goal_relations = set()
        
        for goal in self.goals:
            parts = goal[1:-1].split()
            if not parts:
                continue
            if parts[0] == "dhjk1eyy" and len(parts) == 2:
                self.goal_dhjk1eyy.add(parts[1])
            elif parts[0] == "xk87au5b" and len(parts) == 2:
                self.goal_xk87au5b.add(parts[1])
            elif parts[0] == "x272hpoe" and len(parts) == 3:
                self.goal_relations.add((parts[1], parts[2]))
    
    def __call__(self, node):
        """Compute heuristic value for the given state."""
        state = node.state
        
        # Check if flag is set
        flag_set = "(hxtpdpff)" in state
        
        # Extract objects and their states
        objects = set()
        has_dhjk1eyy = {}
        has_xk87au5b = {}
        has_q71n8mtz = {}
        current_relations = set()
        
        for fact in state:
            # Skip the flag fact
            if fact == "(hxtpdpff)":
                continue
            parts = fact[1:-1].split()
            if not parts:
                continue
            pred = parts[0]
            args = parts[1:]
            for obj in args:
                objects.add(obj)
            if pred == "dhjk1eyy" and len(args) == 1:
                has_dhjk1eyy[args[0]] = True
            elif pred == "xk87au5b" and len(args) == 1:
                has_xk87au5b[args[0]] = True
            elif pred == "q71n8mtz" and len(args) == 1:
                has_q71n8mtz[args[0]] = True
            elif pred == "x272hpoe" and len(args) == 2:
                current_relations.add((args[0], args[1]))
        
        # Initialize missing states for objects not in the dictionaries
        for obj in objects:
            if obj not in has_dhjk1eyy:
                has_dhjk1eyy[obj] = False
            if obj not in has_xk87au5b:
                has_xk87au5b[obj] = False
            if obj not in has_q71n8mtz:
                has_q71n8mtz[obj] = False
        
        # Identify state A objects (have both dhjk1eyy and xk87au5b, not in state B)
        state_A_objects = []
        for obj in objects:
            if has_dhjk1eyy[obj] and has_xk87au5b[obj] and not has_q71n8mtz[obj]:
                state_A_objects.append(obj)
        
        # Find missing goals
        missing_xk87au5b = [obj for obj in self.goal_xk87au5b if not has_xk87au5b.get(obj, False)]
        missing_dhjk1eyy = [obj for obj in self.goal_dhjk1eyy if not has_dhjk1eyy.get(obj, False)]
        missing_relations = [rel for rel in self.goal_relations if rel not in current_relations]
        
        # Extra cost for relations: 1 per relation if source object isn't in state B
        extra_relation_cost = 0
        for (a, b) in missing_relations:
            if not has_q71n8mtz.get(a, False):
                extra_relation_cost += 1
        
        # Check if any state A objects need conversion for missing xk87au5b
        need_action1 = any(obj in state_A_objects for obj in missing_xk87au5b)
        
        # Compute heuristic components
        h_val = 0
        h_val += 2 * len(missing_xk87au5b)  # Convert to state B (1) + apply action2 (1)
        h_val += len(missing_dhjk1eyy)       # Apply action2 or action3 (1)
        h_val += len(missing_relations)       # Apply action3 (1)
        h_val += extra_relation_cost          # Extra conversion cost for relations
        
        # Add cost if flag not set and conversion from state A is needed
        if not flag_set and need_action1:
            h_val += 1
        
        return h_val
