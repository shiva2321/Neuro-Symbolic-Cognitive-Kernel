
try:
    from datasets import load_dataset
    print("Loading tiny_stories sample...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    for i, entry in enumerate(dataset):
        print(f"Entry {i}: {entry['text'][:50]}...")
        if i >= 2: break
    print("SUCCESS")
except Exception as e:
    print(f"FAILED: {e}")
