```bash
clang++ -std=c++11 -o myFilterProgram main.cpp -I./include `pkg-config --cflags --libs opencv4` -framework OpenCL```

./myFilterProgram
```
```bash
brew install opencv
```


```bash
git clone https://github.com/KhronosGroup/OpenCL-Headers.git
```