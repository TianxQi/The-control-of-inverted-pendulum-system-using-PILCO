import subprocess


script_path = 'modul_learn_try.py'

num_runs = 5

for i in range(num_runs):
    print(f"Starting run {i+1} of {num_runs}")

    subprocess.run(['python', script_path])

    print(f"Completed run {i+1} of {num_runs}")

print("All runs completed.")