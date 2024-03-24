import os
import sys

try:
    experiments = os.listdir("results")
except FileNotFoundError:
    print("results Directory not found")
    sys.exit(1)

options = {i: f for i, f in enumerate(experiments, 1)}

if not options:
    print("No experiments found")
    sys.exit(1)

print("Choose an experiment to load:")
for i, f in sorted(options.items()):
    print(f"{i}: {f}")
print()

valid_exp = False
while not valid_exp:
    try:
        exp = int(input("> "))
        if exp in options:
            valid_exp = True
        else:
            print("Invalid experiment number, try again!")
    except ValueError:
        print("Invalid experiment number, try again!")

chosen_exp = options[exp]

options = {i: f for i, f in enumerate(os.listdir(f"results/{chosen_exp}"), 1)}

if not options:
    print("Experiment has no results")
    sys.exit(1)
 
valid_run = False

print("Choose a run to load:")
for i, f in sorted(options.items()):
    print(f"{i}: {f}")
print()

while not valid_run:
    try:
        run = int(input("> "))
        if run in options:
            valid_run = True
        else:
            print("Invalid run number, try again!")
    except ValueError:
        print("Invalid run number, try again!")

chosen_run = options[run]

print("Which dataset would you like to test on?")
datasets = {
    0: "train",
    1: "valid_seen",
    2: "valid_unseen",

    3: "valid_seen (first half)",
    4: "valid_seen (second half)",
    5: "valid_unseen (first half)",
    6: "valid_unseen (second half)",

    7: "valid_seen (commander only)",
    8: "valid_seen (follower only)",

    9: "valid_unseen (commander only)",
    10: "valid_unseen (follower only)",
}

for i, f in sorted(datasets.items()):
    print(f"{i}: {f}")
print()

valid_dataset = False
while not valid_dataset:
    try:
        dataset = int(input("> "))
        if dataset in datasets:
            valid_dataset = True
        else:
            print("Invalid dataset number, try again!")
    except ValueError:
        print("Invalid dataset number, try again!")


if dataset == 0:
    


