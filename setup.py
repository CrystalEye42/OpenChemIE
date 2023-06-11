from setuptools import setup, find_packages

setup(
        name='ChemEScribe',
        version='0.1.0',
        author='Alex Wang',
        author_email='wang7776@mit.edu',
        url='https://github.com/CrystalEye42/ChemEScribe',
        packages=find_packages(),
        package_dir={'chemescribe': 'chemescribe'},
        python_requires='>=3.7',
        install_requires=[
            "numpy",
            "torch>=1.10.0,<2.0",
            "transformers>=4.6.0",
            "layoutparser[effdet]",
            "pdf2image",
            "opencv-python==4.5.5.64",
            "RxnScribe @ git+https://github.com/Ozymandias314/MolDetect.git",
            "MolScribe @ git+https://github.com/thomas0809/MolScribe.git@main#egg=MolScribe"
            ],
        dependency_links=[
            "git+https://github.com/Ozymandias314/MolDetect.git",
            "git+https://github.com/thomas0809/MolScribe.git@main#egg=MolScribe"
        ])

