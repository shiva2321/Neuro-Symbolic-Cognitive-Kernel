
import re
from collections import Counter

log_file = r'd:\NSCK_v1\NSCK_Full_Log_2026-02-01_00-42-08.txt'

tags = Counter()
agents = Counter()
events = Counter()
messages = Counter()

tag_pattern = re.compile(r'\[(.*?)\]')
agent_pattern = re.compile(r'\] (SNAKE|PONG|MAZE) \(')
event_pattern = re.compile(r'\| (AGREE|INTERVENE|VETO|ACTION|LEARN|RULE|DREAM|SLEEP|ERROR|WARNING) \|')

try:
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            # Extract tags
            found_tags = tag_pattern.findall(line)
            for tag in found_tags:
                tags[tag] += 1
            
            # Extract agents
            agent_match = agent_pattern.search(line)
            if agent_match:
                agents[agent_match.group(1)] += 1
            
            # Extract events
            event_match = event_pattern.search(line)
            if event_match:
                events[event_match.group(1)] += 1
            
            # Extract specific interesting messages
            if "Planned path" in line:
                messages["Planned path"] += 1
            if "VSA RESCUE" in line:
                messages["VSA RESCUE"] += 1
            if "Transferring" in line:
                messages["Transferring"] += 1
            if "Saving model" in line:
                messages["Saving model"] += 1

    print("--- Top Tags ---")
    for tag, count in tags.most_common(20):
        print(f"{tag}: {count}")

    print("\n--- Agents ---")
    for agent, count in agents.most_common():
        print(f"{agent}: {count}")

    print("\n--- Events ---")
    for event, count in events.most_common():
        print(f"{event}: {count}")

    print("\n--- Specific Messages ---")
    for msg, count in messages.most_common():
        print(f"{msg}: {count}")

except Exception as e:
    print(f"Error: {e}")
