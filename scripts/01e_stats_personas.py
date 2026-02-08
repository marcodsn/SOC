import os

path = "data/personas/generated"

names = {}
ages = {}
locations = {}

for filename in os.listdir(path):
    if filename.endswith(".txt") and filename.startswith("b_oss120"):
        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            content = f.read()
            lines = content.splitlines()
            for line in lines:
                if "**Name:**" in line:
                    name = line.split("**Name:**")[1].strip()
                    names[name] = names.get(name, 0) + 1
                elif "**Age:**" in line:
                    age = line.split("**Age:**")[1].strip()
                    ages[age] = ages.get(age, 0) + 1
                elif "**Location:**" in line:
                    location = line.split("**Location:**")[1].strip()
                    locations[location] = locations.get(location, 0) + 1

print("Name distribution:")
for name, count in names.items():
    print(f"{name}: {count}")

print("\nAge distribution:")
for age, count in ages.items():
    print(f"{age}: {count}")

print("\nLocation distribution:")
for location, count in locations.items():
    print(f"{location}: {count}")
