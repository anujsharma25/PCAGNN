PCA-GNN Fusion Model This repository contains a Python implementation of an PCA-GNN Fusion model, which combines Principal Component Analysis (PCA) with Graph Neural Networks (GNN) for graph-based classification tasks. The model integrates PCA-derived data into the GNN's message-passing mechanism, using attention scores to guide node embedding updates. Overview The code implements a hybrid machine learning model that leverages:

PCA: reduce dimensions of data (especially effective in large dimensional cases). GNN: Performs message passing with attention-based aggregation, incorporating support vector contributions. Loss Function: Combines cross-entropy loss, hinge loss, L2 regularization, and a gamma penalty for stable training.

The model is designed to run in a Pyodide environment (e.g., browser-based Python) but is also compatible with standard Python environments. It uses PyTorch for neural network operations, scikit-learn for PCA, and NumPy for numerical computations. Requirements To run the code, ensure you have the following dependencies installed:

Python 3.8 or higher PyTorch (torch>=1.8.0) NumPy (numpy>=1.22.0) scikit-learn (scikit-learn>=1.0.0) SciPy (scipy>=1.7.0)

For Pyodide compatibility, the code includes checks for the Emscripten platform to handle asynchronous execution in a browser environment. Installation

Clone the repository: git clone cd

Create a virtual environment (optional but recommended): python -m venv venv source venv/bin/activate # On Windows: venv\Scripts\activate

Install dependencies: pip install torch numpy scikit-learn scipy

Ensure PyTorch is compatible with your system (CPU/GPU). Visit PyTorch's official site for installation instructions.

Usage The code is contained in a single Python script and c file plug-in. It includes a Config class for hyperparameters, an PCAGNNFusion class for the model, and a train_and_predict function for training and inference. Running the Code

Prepare your data:The model expects:

X: Node feature matrix (shape: [num_nodes, input_dim]). y: Node labels (shape: [num_nodes]). G: Edge index array (shape: [2, num_edges]) representing the graph structure. train_mask: Boolean array indicating training nodes (shape: [num_nodes]). num_classes: Number of classes for classification.

The provided code includes a toy example with random data for demonstration: num_nodes, input_dim, num_classes = 100, 16, 3 X = np.random.randn(num_nodes, input_dim) y = np.random.randint(0, num_classes, num_nodes) train_mask = np.random.choice([True, False], num_nodes, p=[0.8, 0.2]) G = np.random.randint(0, num_nodes, (2, num_nodes * 2))

Run the script and C file to create dependency;

The script will:

Initialize the model with the specified configuration. Extract data using PCA. Train the GNN for 100 epochs, printing the loss every 10 epochs. Output predictions for all nodes.

Example Output Epoch 0, Loss: 1.2345 Epoch 10, Loss: 0.9876 ... Epoch 90, Loss: 0.4567 Predictions: [0 2 1 ...]

Model Architecture The PCAGNNFusion model operates in three phases:

For C++ file,
g++ -I/path/to/libtorch/include -I/path/to/libtorch/include/torch/csrc/api/include \
    -L/path/to/libtorch/lib main.cpp -ltorch -lc10 -o pgf_model \
    -Wl,-rpath,/path/to/libtorch/lib -std=c++17


Notes

Compatibility: The code includes a check for platform.system() == "Emscripten" to support browser-based execution. In standard Python environments, it uses asyncio.run(main()). Data Requirements: The model assumes graph-structured data with node features and edge indices. Replace the toy data with your own dataset for practical use. Performance: The model’s performance depends on the quality of the input graph and features. Tune hyperparameters in Config for better results. Limitations: The PCA training step can be computationally expensive for large datasets. 
