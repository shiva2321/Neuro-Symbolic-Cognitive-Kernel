"""
🚀 Quick Experiment Launcher
Select and run neuromorphic experiments with one command
"""

import sys


def print_banner():
    print("\n" + "="*80)
    print("🧠 NEUROMORPHIC EXPERIMENT QUICK LAUNCHER")
    print("="*80 + "\n")


def print_menu():
    print("Available Experiments:")
    print("  [1] Pavlov's Dog - Classical Conditioning (⭐ Easy)")
    print("  [2] Sequence Learning - Temporal Patterns (⭐⭐ Medium)")
    print("  [3] XOR Problem - Non-Linear Classification (⭐⭐⭐ Hard)")
    print("  [4] Unified Dashboard - All Experiments")
    print("  [5] Learning Dashboard - Interactive Pavlov")
    print("  [6] 🌟 UNIFIED LEARNER - ONE Network, All Skills! (Recommended)")
    print("  [0] Exit\n")


def run_experiment(choice):
    """Run the selected experiment."""
    if choice == '1':
        print("\n🔬 Launching Pavlov's Dog Experiment...")
        from experiments.pavlov_experiment import main
        main()

    elif choice == '2':
        print("\n🔬 Launching Sequence Learning Experiment...")
        print("💡 TIP: This may need 300-500 epochs to learn well.")

        # Ask for custom epochs
        try:
            epochs_input = input("\nEnter epochs (press ENTER for default 100): ").strip()
            if epochs_input:
                epochs = int(epochs_input)
            else:
                epochs = 100

            from experiments.sequence_experiment import SequenceLearningExperiment
            experiment = SequenceLearningExperiment()
            experiment.run_training(max_epochs=epochs)

        except ValueError:
            print("⚠ Invalid number, using default 100 epochs")
            from experiments.sequence_experiment import main
            main()

    elif choice == '3':
        print("\n🔬 Launching XOR Problem Experiment...")
        print("💡 TIP: This may need 400-800 epochs to solve completely.")

        # Ask for custom epochs
        try:
            epochs_input = input("\nEnter epochs (press ENTER for default 200): ").strip()
            if epochs_input:
                epochs = int(epochs_input)
            else:
                epochs = 200

            from experiments.xor_experiment import XORExperiment
            experiment = XORExperiment()
            experiment.run_training(max_epochs=epochs, target_accuracy=100.0)

        except ValueError:
            print("⚠ Invalid number, using default 200 epochs")
            from experiments.xor_experiment import main
            main()

    elif choice == '4':
        print("\n📊 Launching Unified Experiment Dashboard...")
        print("⚠ Note: experiment_dashboard module not found, skipping...")

    elif choice == '5':
        print("\n📊 Launching Learning Dashboard (Pavlov)...")
        from tools.learning_dashboard import main
        main()

    elif choice == '6':
        print("\n🌟 Launching Unified Learner (ONE Network for All Tasks)...")
        print("💡 This demonstrates continuous learning with teacher forcing!")
        print("   The network will learn Pavlov → Sequence → XOR on one brain.\n")
        from experiments.unified_learner import main
        main()

    elif choice == '0':
        print("\n👋 Goodbye!\n")
        return False

    else:
        print("\n⚠ Invalid choice!")

    return True


def main():
    """Main launcher loop."""
    print_banner()

    # Check if command-line argument provided
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        run_experiment(choice)
        return

    # Interactive mode
    while True:
        print_menu()
        choice = input("Select experiment: ").strip()

        should_continue = run_experiment(choice)
        if not should_continue:
            break

        print("\n" + "="*80 + "\n")
        input("Press ENTER to return to menu...")
        print("\n" * 2)


if __name__ == "__main__":
    main()

