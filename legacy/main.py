# [F-memory/R3/C3] Entry point — demonstrates full LM lifecycle
# Run: python main.py

from src.contract.api import LivingMemory


def main():
    print("Living Memory — LM")
    print("==================")

    with LivingMemory(db_path="demo.db") as lm:

        # Backbone — permanent identity, never decays
        lm.remember("backbone", "User is Demos. Building LIV — structural AI dev platform.", tags=["identity"])
        lm.remember("backbone", "Core principle: code self-organizes under Living Architecture.", tags=["identity"])

        # Dynamic branches — created on first use
        lm.remember("goals", "Ship living memory as open-source library for agent memory persistence.")
        lm.remember("goals", "Living memory goal: integrate LM into LIV as default agent memory system.")
        lm.remember("goals", "Living memory goal: validate multi-year memory persistence with agent workloads.")
        lm.remember("beliefs", "Agent memory must be context-aware and semantically indexed not time-indexed.")
        lm.remember("beliefs", "Memory compression should self-trigger via phi ratio not manual management.")
        lm.remember("threads", "Open thread: benchmark TF-IDF cosine retrieval against sentence-transformers.")
        lm.remember("threads", "Open thread: define agent session boundary protocol for root OLS compression.")

        # Warm corpus — queries update access counts and IDF
        lm._tree._rebuild_idf()

        # Query
        print("\n--- recall: 'goals for living memory' ---")
        print(lm.recall("goals for living memory"))

        print("\n--- recall: 'open threads and benchmarks' ---")
        print(lm.recall("open threads and benchmarks"))

        # Status
        print("\n--- status ---")
        import json
        print(json.dumps(lm.status(), indent=2))

        # Root OLS (cold branch compression)
        compressed = lm.compress()
        print(f"\n--- root OLS ran, compressed branches: {compressed} ---")

        # Export
        path = lm.export("demo_export.json")
        print(f"\n--- exported to {path} ---")

    print("\nDone.")


if __name__ == "__main__":
    import os
    if os.path.exists("demo.db"):
        os.remove("demo.db")
    main()
