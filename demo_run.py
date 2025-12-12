"""
Enhanced Demo Script
Demonstrates all features of the improved Game Theory Simulator
"""

from game_theory_core import NormalFormGame, ExtensiveFormGame
import numpy as np

def demo_prisoners_dilemma():
    """Demonstrate analysis of Prisoner's Dilemma."""
    print("\n" + "="*80)
    print("DEMO 1: PRISONER'S DILEMMA")
    print("="*80)
    
    A = np.array([[-1, -4], [0, -3]])
    B = np.array([[-1, 0], [-4, -3]])
    labels = (['Cooperate', 'Defect'], ['Cooperate', 'Defect'])
    game = NormalFormGame(A, B, labels, "Prisoner's Dilemma")
    
    # Show the game
    game.pretty_print()
    
    # Comprehensive analysis
    game.comprehensive_analysis()
    
    # Generate visualizations
    game.plot_payoff_heatmap('demo_prisoners_dilemma_heatmap.png')
    
    print("\n📌 KEY INSIGHTS:")
    print("   • Nash equilibrium (Defect, Defect) is not Pareto optimal")
    print("   • Both players would be better off at (Cooperate, Cooperate)")
    print("   • This illustrates the conflict between individual and collective rationality")
    print("   • Cooperate is a strictly dominated strategy for both players")

def demo_battle_of_sexes():
    """Demonstrate Battle of the Sexes with multiple equilibria."""
    print("\n" + "="*80)
    print("DEMO 2: BATTLE OF THE SEXES")
    print("="*80)
    
    A = np.array([[2, 0], [0, 1]])
    B = np.array([[1, 0], [0, 2]])
    labels = (['Opera', 'Football'], ['Opera', 'Football'])
    game = NormalFormGame(A, B, labels, "Battle of the Sexes")
    
    game.pretty_print()
    game.comprehensive_analysis()
    game.plot_payoff_heatmap('demo_battle_of_sexes_heatmap.png')
    
    print("\n📌 KEY INSIGHTS:")
    print("   • Two pure Nash equilibria: (Opera, Opera) and (Football, Football)")
    print("   • Also has a mixed strategy Nash equilibrium")
    print("   • Coordination problem: players prefer different outcomes")
    print("   • Both pure equilibria are Pareto efficient")

def demo_matching_pennies():
    """Demonstrate zero-sum game with no pure Nash equilibrium."""
    print("\n" + "="*80)
    print("DEMO 3: MATCHING PENNIES")
    print("="*80)
    
    A = np.array([[1, -1], [-1, 1]])
    B = -A
    labels = (['Heads', 'Tails'], ['Heads', 'Tails'])
    game = NormalFormGame(A, B, labels, "Matching Pennies")
    
    game.pretty_print()
    game.comprehensive_analysis()
    game.plot_payoff_heatmap('demo_matching_pennies_heatmap.png')
    
    print("\n📌 KEY INSIGHTS:")
    print("   • Zero-sum game: one player's gain is the other's loss")
    print("   • No pure strategy Nash equilibrium exists")
    print("   • Unique mixed strategy Nash equilibrium: both play 50-50")
    print("   • Expected payoff is 0 for both players at equilibrium")

def demo_hawk_dove():
    """Demonstrate Hawk-Dove game with asymmetric equilibria."""
    print("\n" + "="*80)
    print("DEMO 4: HAWK-DOVE GAME")
    print("="*80)
    
    A = np.array([[-2, 3], [0, 1]])
    B = np.array([[-2, 0], [3, 1]])
    labels = (['Hawk', 'Dove'], ['Hawk', 'Dove'])
    game = NormalFormGame(A, B, labels, "Hawk-Dove Game")
    
    game.pretty_print()
    game.comprehensive_analysis()
    game.plot_payoff_heatmap('demo_hawk_dove_heatmap.png')
    
    print("\n📌 KEY INSIGHTS:")
    print("   • Two asymmetric pure Nash equilibria")
    print("   • (Hawk, Dove) and (Dove, Hawk)")
    print("   • Also has a mixed strategy equilibrium")
    print("   • Models conflict escalation and backing down")

def demo_stag_hunt():
    """Demonstrate Stag Hunt with payoff vs risk dominance."""
    print("\n" + "="*80)
    print("DEMO 5: STAG HUNT")
    print("="*80)
    
    A = np.array([[4, 0], [3, 3]])
    B = np.array([[4, 3], [0, 3]])
    labels = (['Stag', 'Hare'], ['Stag', 'Hare'])
    game = NormalFormGame(A, B, labels, "Stag Hunt")
    
    game.pretty_print()
    game.comprehensive_analysis()
    game.plot_payoff_heatmap('demo_stag_hunt_heatmap.png')
    
    print("\n📌 KEY INSIGHTS:")
    print("   • Two pure Nash equilibria: (Stag, Stag) and (Hare, Hare)")
    print("   • (Stag, Stag) is payoff-dominant: higher payoffs for both")
    print("   • (Hare, Hare) is risk-dominant: safer choice")
    print("   • Illustrates coordination problems with risk")

def demo_extensive_form():
    """Demonstrate extensive form game analysis."""
    print("\n" + "="*80)
    print("DEMO 6: EXTENSIVE FORM - SEQUENTIAL ENTRY GAME")
    print("="*80)
    
    tree = {
        'root': {'player': 0, 'actions': {'Enter': 'entrant_enter', 'Stay_Out': 'stay_out'}},
        'entrant_enter': {'player': 1, 'actions': {'Fight': 'fight', 'Accommodate': 'accommodate'}},
        'stay_out': {'payoff': (0, 2)},
        'fight': {'payoff': (-1, -1)},
        'accommodate': {'payoff': (1, 1)},
    }
    
    game = ExtensiveFormGame(tree, ['Entrant', 'Incumbent'], "Sequential Entry Game")
    
    game.comprehensive_analysis()
    game.plot_tree('demo_entry_game_tree.png')
    
    print("\n📌 KEY INSIGHTS:")
    print("   • Backward induction reveals the equilibrium path")
    print("   • Incumbent's threat to 'Fight' is not credible")
    print("   • SPNE: Entrant enters, Incumbent accommodates")
    print("   • Sequential games allow for credible commitments")

def demo_custom_analysis():
    """Demonstrate detailed strategic analysis."""
    print("\n" + "="*80)
    print("DEMO 7: DETAILED STRATEGIC ANALYSIS")
    print("="*80)
    
    # Create a 3x3 game with interesting properties
    A = np.array([
        [3, 0, 5],
        [2, 4, 3],
        [1, 3, 2]
    ])
    B = np.array([
        [3, 2, 1],
        [0, 4, 3],
        [5, 3, 2]
    ])
    labels = (
        ['Strategy A1', 'Strategy A2', 'Strategy A3'],
        ['Strategy B1', 'Strategy B2', 'Strategy B3']
    )
    game = NormalFormGame(A, B, labels, "Complex 3x3 Game")
    
    game.pretty_print()
    
    # Show best responses
    print("\n" + "-"*80)
    print("BEST RESPONSE ANALYSIS")
    print("-"*80)
    
    print("\nPlayer 0's Best Responses:")
    for j in range(game.n):
        opp_strat = [1 if k == j else 0 for k in range(game.n)]
        br = game.best_response(0, opp_strat)
        br_labels = [game.labels[0][i] for i in br]
        payoffs = [game.A[i, j] for i in br]
        print(f"  Against {game.labels[1][j]}: {br_labels} (payoff: {payoffs})")
    
    print("\nPlayer 1's Best Responses:")
    for i in range(game.m):
        opp_strat = [1 if k == i else 0 for k in range(game.m)]
        br = game.best_response(1, opp_strat)
        br_labels = [game.labels[1][j] for j in br]
        payoffs = [game.B[i, j] for j in br]
        print(f"  Against {game.labels[0][i]}: {br_labels} (payoff: {payoffs})")
    
    # Pareto analysis
    print("\n" + "-"*80)
    print("PARETO EFFICIENCY ANALYSIS")
    print("-"*80)
    
    pareto_outcomes = game.pareto_efficient_outcomes()
    print(f"\nPareto Efficient Outcomes ({len(pareto_outcomes)} total):")
    for i, j in pareto_outcomes:
        print(f"  ({game.labels[0][i]}, {game.labels[1][j]})")
        print(f"    Payoffs: ({game.A[i,j]:.2f}, {game.B[i,j]:.2f}), "
              f"Total Welfare: {game.social_welfare(i,j):.2f}")
    
    # Social optimum
    i_opt, j_opt = game.social_optimum()
    print(f"\nSocial Optimum:")
    print(f"  ({game.labels[0][i_opt]}, {game.labels[1][j_opt]})")
    print(f"  Maximizes total welfare: {game.social_welfare(i_opt, j_opt):.2f}")
    
    game.plot_payoff_heatmap('demo_complex_3x3_heatmap.png')

def demo_mixed_strategy_calculation():
    """Demonstrate mixed strategy equilibrium calculation."""
    print("\n" + "="*80)
    print("DEMO 8: MIXED STRATEGY EQUILIBRIUM CALCULATION")
    print("="*80)
    
    # Battle of the Sexes - has interesting mixed equilibrium
    A = np.array([[2, 0], [0, 1]])
    B = np.array([[1, 0], [0, 2]])
    labels = (['Opera', 'Football'], ['Opera', 'Football'])
    game = NormalFormGame(A, B, labels, "Battle of the Sexes")
    
    game.pretty_print()
    
    print("\n" + "-"*80)
    print("EQUILIBRIUM ANALYSIS")
    print("-"*80)
    
    # Pure equilibria
    pure_ne = game.pure_nash_equilibria()
    print(f"\nPure Nash Equilibria ({len(pure_ne)} found):")
    for i, j in pure_ne:
        u0, u1 = game.A[i, j], game.B[i, j]
        print(f"  ({game.labels[0][i]}, {game.labels[1][j]}) → Payoffs: ({u0:.2f}, {u1:.2f})")
    
    # Mixed equilibria
    mixed_ne = game.support_enumeration_mixed_nash()
    print(f"\nMixed Nash Equilibria ({len(mixed_ne)} found):")
    for idx, (sig0, sig1) in enumerate(mixed_ne, 1):
        print(f"\n  Equilibrium {idx}:")
        for i, prob in enumerate(sig0):
            if prob > 1e-6:
                print(f"    Player 0 plays {game.labels[0][i]} with probability {prob:.4f}")
        for j, prob in enumerate(sig1):
            if prob > 1e-6:
                print(f"    Player 1 plays {game.labels[1][j]} with probability {prob:.4f}")
        u0, u1 = game.expected_payoffs(sig0, sig1)
        print(f"    Expected payoffs: ({u0:.4f}, {u1:.4f})")

def demo_iterative_elimination():
    """Demonstrate iterative elimination of dominated strategies."""
    print("\n" + "="*80)
    print("DEMO 9: ITERATIVE ELIMINATION OF DOMINATED STRATEGIES")
    print("="*80)
    
    # Create a game with dominated strategies
    A = np.array([
        [2, 0, 1],
        [1, 1, 1],
        [3, 2, 0]
    ])
    B = np.array([
        [2, 1, 3],
        [0, 1, 2],
        [1, 1, 0]
    ])
    labels = (
        ['Top', 'Middle', 'Bottom'],
        ['Left', 'Center', 'Right']
    )
    game = NormalFormGame(A, B, labels, "Game with Dominated Strategies")
    
    print("\nOriginal Game:")
    game.pretty_print()
    
    # Check for dominated strategies
    print("\n" + "-"*80)
    print("DOMINATED STRATEGY ANALYSIS")
    print("-"*80)
    
    dom0_strict = game.strictly_dominated_pure(0)
    dom1_strict = game.strictly_dominated_pure(1)
    
    if dom0_strict:
        print(f"\nPlayer 0 strictly dominated strategies:")
        for i in dom0_strict:
            print(f"  • {game.labels[0][i]}")
    if dom1_strict:
        print(f"\nPlayer 1 strictly dominated strategies:")
        for j in dom1_strict:
            print(f"  • {game.labels[1][j]}")
    
    # Iterative elimination
    print("\n" + "-"*80)
    print("ITERATIVE ELIMINATION PROCESS")
    print("-"*80)
    
    (remaining_rows, remaining_cols), reduced_game = game.iterative_elimination_strictly_dominated()
    
    print(f"\nRemaining strategies after elimination:")
    print(f"  Player 0: {[game.labels[0][i] for i in remaining_rows]}")
    print(f"  Player 1: {[game.labels[1][j] for j in remaining_cols]}")
    
    if len(remaining_rows) < game.m or len(remaining_cols) < game.n:
        print("\nReduced Game:")
        reduced_game.pretty_print()

def run_all_demos():
    """Run all demonstration scripts."""
    print("\n" + "="*80)
    print(" "*20 + "GAME THEORY SIMULATOR")
    print(" "*15 + "Comprehensive Feature Demonstration")
    print("="*80)
    
    demos = [
        ("Prisoner's Dilemma Analysis", demo_prisoners_dilemma),
        ("Battle of the Sexes - Multiple Equilibria", demo_battle_of_sexes),
        ("Matching Pennies - Mixed Strategies", demo_matching_pennies),
        ("Hawk-Dove Game - Asymmetric Equilibria", demo_hawk_dove),
        ("Stag Hunt - Payoff vs Risk Dominance", demo_stag_hunt),
        ("Extensive Form Games & Backward Induction", demo_extensive_form),
        ("Detailed Strategic Analysis", demo_custom_analysis),
        ("Mixed Strategy Calculations", demo_mixed_strategy_calculation),
        ("Iterative Elimination", demo_iterative_elimination),
    ]
    
    print("\nAvailable Demonstrations:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    print("  A. Run All Demos")
    print("  Q. Quit")
    
    choice = input("\nSelect a demo to run (or press Enter for all): ").strip().upper()
    
    if choice == '' or choice == 'A':
        for name, demo_func in demos:
            print(f"\n{'='*80}")
            print(f"Running: {name}")
            print(f"{'='*80}")
            demo_func()
            input("\nPress Enter to continue to next demo...")
    elif choice == 'Q':
        return
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(demos):
                demos[idx][1]()
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input.")
    
    print("\n" + "="*80)
    print("Demo completed! Check generated PNG files for visualizations.")
    print("="*80)

if __name__ == '__main__':
    run_all_demos()