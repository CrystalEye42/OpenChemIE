from setuptools import setup

setup(
        name='ChemEScribe',
        version='0.1.0',
        author='Alex Wang',
        author_email='wang7776@mit.edu',
        url='https://github.com/CrystalEye42/ChemEScribe',
        packages=['chemescribe'],
        package_dir={'chemescribe', 'chemescribe'},
        python_requires='>=3.7',
        install_requires=[
            "numpy",
            "torch>=1.10.0,<2.0",
            "transformers>=4.6.0",
            "layoutparser",
            "detectron2",
            "pdf2image",
            "RxnScribe @ git+https://github.com/thomas0809/RxnScribe.git@main#egg=RxnScribe",
            "MolScribe @ git+https://github.com/thomas0809/MolScribe.git@main#egg=MolScribe",
            ],
        dependency_links=[
            "git+https://github.com/thomas0809/RxnScribe.git@main#egg=RxnScribe",
        ])

