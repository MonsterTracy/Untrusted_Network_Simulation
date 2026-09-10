from setuptools import setup, find_packages

setup(
    name="werewolf",
    version="0.1",
    description="Multi-agent Theory-of-Mind reasoning in Werewolf.",
    keywords="werewolf, multi-agent, theory-of-mind",
    packages=find_packages(),
    py_modules=["run_random"],
    entry_points={"console_scripts": ["uns=werewolf.cli:main"]},
    python_requires=">=3.10",
    install_requires=[
        "gymnasium",
        "numpy>=1.24,<3.0",
        "openai>=3.10.0",
        "pydantic>=2.10.4",
        "python-dotenv>=1.0.0",
        "PyYAML>=6.0.2",
        "tenacity>=9.0.0",
        "tiktoken>=0.7.0",
    ],
    extras_require={
        "tom": [
            "torch>=2.0.0",
            "transformers>=4.47.1",
        ],
        "dev": [
            "pytest",
        ],
    },
)
