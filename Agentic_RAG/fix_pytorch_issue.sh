# Fix for PyTorch meta tensor issue with sentence-transformers
# Run this if you encounter "Cannot copy out of meta tensor" error

# Option 1: Update to compatible versions
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install --upgrade sentence-transformers

# Option 2: If Option 1 doesn't work, use specific compatible versions
# pip install torch==2.0.1
# pip install sentence-transformers==2.2.2

# Then restart Django
