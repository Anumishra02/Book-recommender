from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

REPO_NAME = "ML based Books Recommender System"
AUTHOR_USER_NAME = "Anu Mishra"
SRC_REPO = "books_recommender"
LIST_OF_REQUIREMENTS = []


setup(
    name=SRC_REPO,
    version="0.0.1",
    author="Anu Mishra",
    description="A small local package for Machine Learning based Books Recommendations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Anumishra02/Book-recommender",
    # url="https://github.com/entbappy/ML-Based-Book-recommender-System",
    author_email="anumishra555555@gmail.com",
    packages=find_packages(),
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)
