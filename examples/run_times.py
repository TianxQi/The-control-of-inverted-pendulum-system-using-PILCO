import subprocess

for i in range(3):
    subprocess.run(["python", "moduldata_train.py"])
    print('已运行',i+1)

