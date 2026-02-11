import json
import os

path = "data/personas/generated"

names = {}
ages = {}
locations = {}

for filename in os.listdir(path):
    if filename.endswith(".jsonl") and filename.startswith("b_personas"):
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()
            for line in lines:
                data = json.loads(line)

                name = data.get("meta", {}).get("name", "Unknown")
                names[name] = names.get(name, 0) + 1

                age = data.get("meta", {}).get("age", "Unknown")
                ages[age] = ages.get(age, 0) + 1

                location = data.get("meta", {}).get("region", "Unknown")
                locations[location] = locations.get(location, 0) + 1

# Sort the distributions by count
names = dict(sorted(names.items(), key=lambda item: item[1], reverse=True))
ages = dict(sorted(ages.items(), key=lambda item: item[1], reverse=True))
locations = dict(sorted(locations.items(), key=lambda item: item[1], reverse=True))

print("Name distribution:")
for name, count in names.items():
    print(f"{name}: {count}")

print("\nAge distribution:")
for age, count in ages.items():
    print(f"{age}: {count}")

print("\nLocation distribution:")
for location, count in locations.items():
    print(f"{location}: {count}")

print("\nTotal personas analyzed:", sum(names.values()))
