from setuptools import setup, find_packages

setup(
    name="hieroglyph_processor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-multipart",
        "opencv-python-headless",
        "numpy",
        "torch",
        "tensorflow",
        "scikit-learn",
        "segment-anything",
        "pandas",
        "pyyaml",
        "requests",
        "tqdm",
        "openai"
    ],
    entry_points={
        'console_scripts': [
            'hieroglyph-api = translation_service.api_server:main',
            'hieroglyph-batch = translation_service.batch_processor:main',
            'download-models = translation_service.model_downloader:main'
        ]
    }
)