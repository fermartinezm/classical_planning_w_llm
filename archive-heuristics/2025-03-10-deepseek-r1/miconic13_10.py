from fnmatch import fnmatch
from heuristics.heuristic_base import Heuristic

def get_parts(fact):
    """Extract components of a PDDL fact string."""
    return fact[1:-1].split()

def match(fact, *args):
    """Check if a fact matches a pattern with wildcards."""
    parts = get_parts(fact)
    return all(fnmatch(part, arg) for part, arg in zip(parts, args))

class Miconic13Heuristic(Heuristic):
    """
    A domain-dependent heuristic for the Miconic domain.

    # Summary
    Estimates the number of actions required to serve all passengers by considering the current lift position, 
    each passenger's origin and destination, and their boarded/served status. The heuristic accounts for lift 
    movements, board actions, and depart actions.

    # Assumptions
    - Each passenger must be boarded at their origin and departed at their destination.
    - The lift can move directly between any two floors in one action due to the 'above' hierarchy.
    - The heuristic does not need to be admissible, focusing on reducing expanded nodes in greedy search.

    # Heuristic Initialization
    - Extracts static origin and destination floors for each passenger from the problem's static facts.

    # Step-By-Step Thinking for Computing Heuristic
    1. Determine the current lift position from the state.
    2. For each passenger:
        a. If already served, skip.
        b. If boarded, add actions for moving to destination (if needed) and departing.
        c. If not boarded, add actions for moving to origin (if needed), boarding, moving to destination (if needed), and departing.
    3. Sum all actions and movements to get the heuristic value.
    """

    def __init__(self, task):
        self.origin = {}
        self.destin = {}
        for fact in task.static:
            parts = get_parts(fact)
            if parts[0] == 'origin' and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.origin[p] = f
            elif parts[0] == 'destin' and len(parts) == 3:
                p, f = parts[1], parts[2]
                self.destin[p] = f

    def __call__(self, node):
        state = node.state
        current_lift = None
        for fact in state:
            if match(fact, 'lift-at', '*'):
                current_lift = get_parts(fact)[1]
                break
        
        h = 0
        for p in self.origin:
            if f'(served {p})' in state:
                continue
            
            if f'(boarded {p})' in state:
                dest = self.destin[p]
                if current_lift != dest:
                    h += 1  # Move to destination
                h += 1  # Depart action
            else:
                orig = self.origin[p]
                dest = self.destin[p]
                if current_lift != orig:
                    h += 1  # Move to origin
                h += 1  # Board action
                if orig != dest:
                    h += 1  # Move to destination
                h += 1  # Depart action
        return h
