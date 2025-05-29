# cs179_final_project
## CPU Demo
```
# On Ubuntu with a monitor
git clone https://github.com/AndyLinGitHub/cs179_final_project.git
cd cs179_final_project
sudo apt install freeglut3 freeglut3-dev
python3 -m venv nca
source nca/bin/activate
pip install -r requirements.txt
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
python cpu_demo.py
```
