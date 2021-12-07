import setuptools

setuptools.setup(
    name="clearformer",
    version="0.1",
    description="Creating explainable embeddings",
    url="#",
    author="Jonathan Rystroem",
    install_requires=[
        "scikit-learn==1.0.1",
        "numpy>=1.20.0",
        "bertopic==0.9.3",
        "matplotlib",
    ],
    author_email="jhvithamar@gmail.com",
    packages=setuptools.find_packages(),
    zip_safe=False,
)

