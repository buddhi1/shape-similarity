# Artifact Appendix

## **Abstract**
This artifact appendix provides the source code and dataset information to replicate the experiments demonstrated in Chapter 3. The artifact contains bash scripts to conveniently execute the test cases over the datasets.   

## **Artifact Check-list (Meta Information)**

- **Program:** C++ language
- **Compilation:** PGC++ compiler
- **Dataset:** The artifact requires a dataset of coordinates of polygons (at least a polygon pair). The dataset used for the experiments is in the `data` folder (`poly_data.zip`).
- **Run-time environment:** Ubuntu 22.04
- **Execution:** The CPU and GPU need to be free while executing the experiments. Otherwise, it can affect the speedup of the parallel algorithm.
- **Metrics:** The comparisons are based on the speedup metric. The formula for other metrics is mentioned in Chapter 3.
- **Output:** If there is any intersection between the input polygon pair, the output is a plain text file that contains the coordinates of the overlapping region of that polygon pair. It is saved in the `results/` folder. In cases where the output contains a collection of polygons, their coordinates are split using ';'.
- **Experiments:** Run the GPU algorithm, run Foster's sequential algorithm, and record execution times in the given spreadsheet to calculate speedup and generate graphs.
- **Required disk space (approximately):** 500MB
- **Time needed to prepare workflow (approximately):** 20 minutes
- **Time needed to complete experiments (approximately):** 30 minutes
- **Publicly available:** [GitHub](https://github.com/buddhi1/segment-tree-based-clipping)

## **Description**

**How to access:**
The source code is available in the main branch of the following GitHub repository: [https://github.com/buddhi1/segment-tree-based-clipping](https://github.com/buddhi1/segment-tree-based-clipping)

**Hardware dependencies:**
This artifact requires a Linux x86 machine with an Nvidia GPU (tested on `compute_70`, `compute_75`, `compute_80`, and `compute_86` architectures). To replicate the experiments over real-world datasets, 500MB of free space is approximately required.

The specifications of the machine used are as follows:
- Processor: Intel® Xeon(R) Silver 4210R CPU @ 2.40GHz
- Memory: 64GB
- GPU: NVIDIA Quadro RTX 5000 with 16GB memory

**Software dependencies:**
A Linux-based OS is required to build this artifact. The GPU uses Nvidia driver version 535.
- GNU make
- C++ 11
- OpenMP
- OpenACC
- Thrust
- unzip

**Dataset:**
There are multiple real-world polygon data files. Each file consists of the (x, y) coordinates of a polygon. The first line represents the total number of vertices in the polygon. The data file path is `data/poly_data.zip`.

## **Installation**

1. Clone the repository:
    ```sh
    $ git clone https://github.com/buddhi1/segment-tree-based-clipping.git
    ```
2. Change to the project directory:
    ```sh
    $ cd segment-tree-based-clipping
    ```
3. Compile the code:
    ```sh
    $ make
    ```

The `make` file reads the `cuda` path from the `$LD_LIBRARY_PATH` variable to link the `thrust` library. If the path is not correctly found, replace the `LIBCUDA` variable in the `make` file with the correct path.

Use the following commands to unzip the data:
1. Change to the data directory:
    ```sh
    $ cd data/
    ```
2. Unzip the dataset:
    ```sh
    $ unzip poly_data.zip
    ```
3. Return to the main directory:
    ```sh
    $ cd ..
    ```

Now the `data` folder should have all extracted data files.

**How to make the project:**

1. Clean build:
    ```sh
    $ make clean
    ```
2. Compile for CPU version:
    ```sh
    $ make
    ```
3. Compile for Hybrid (CPU+GPU) version:
    ```sh
    $ make gpu
    ```

**Debug mode:**

To enter debugging mode, pass `D=0` when making. To start saving the results, pass `S=0` when making.
Example:

1. Clean:
    ```sh
    $ make clean
    ```
2. Compile for CPU version in debug mode saving results:
    ```sh
    $ make D=0 S=0
    ```

## **Experimental Workflow**

**How to run:**

Use the following template to find the intersection between a pair of polygons:
```sh
$ OMP_NUM_THREADS=<core count> bin/./seg \
    <base polygon file> \
    <overlay polygon file> \
    <result polygon path>
```

Demonstrated experiments have 3 cases. For convenience, we have included two bash scripts to perform pairwise and one-to-many polygon clipping. 

### **Polygon Clipping**

#### **Multi-core Experiments**

1. Clean:
    ```sh
    $ make clean
    ```
2. Make:
    ```sh
    $ make
    ```
3. Run experiments:
    ```sh
    $ sh run-omp.sh
    ```

#### **Many-core Experiments**

1. Clean:
    ```sh
    $ make clean
    ```
2. Make:
    ```sh
    $ make gpu
    ```
3. Run experiments:
    ```sh
    $ sh run-omp.sh
    ```

### **One-to-many Polygon Clipping**

#### **Multi-core Experiments**

1. Clean:
    ```sh
    $ make clean
    ```
2. Make:
    ```sh
    $ make
    ```
3. Run experiments:
    ```sh
    $ sh run-1-to-m.sh
    ```

#### **Many-core Experiments**

1. Clean:
    ```sh
    $ make clean
    ```
2. Make:
    ```sh
    $ make gpu
    ```
3. Run experiments:
    ```sh
    $ sh run-1-to-m.sh
    ```

**Save results:**
Use `S=0` when making to project. All results are saved in the `results` folder.

