# John progress handover notes for stefan/JackS

- Private clone of PrimAITE into our org github.
- Added to AEGIS pyproject.toml as dependency.
- When doing poetry install, you'll get the PEP 517 build error with gym library installation... this is something to do with the version of setuptools I think.
- It looks like there are some extras (called 'dev' - see the primaite readme) which are by default not installed but should be: [See here](https://stackoverflow.com/questions/60971502/python-poetry-how-to-install-optional-dependencies).
- I was working on figuring out how to get the 'dev' group of extra dependencies to install, as you'll see in our the pyproject.toml.
    - You can probs sort this an easier way by copy pasting these dependencies from the dev group into real dependencies in our 'fork' of primaite... unfortunately I've ran out of time.