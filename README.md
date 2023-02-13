InsideGPT
=========

[![Open Source Love](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/) [![GitHub license](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/Naereen/StrapDown.js/blob/master/LICENSE)

InsideGPT is a project that allows you to explore document for sending theses with embbedings and use OpenAI's GPT-3 language model for generating text.

Getting Started
---------------

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

You will need to have the following tools installed on your system to be able to run the project:

-   Python 3.7 or higher
-   Poetry (<https://python-poetry.org/>)
-   Streamlit (<https://docs.streamlit.io/getting_started.html>)

### Installing

To install the dependencies, follow these steps:

1.  Clone the repository to your local machine:



`git clone https://github.com/RemiPelloux/InsideGPT`

1.  Change into the newly created directory:



`cd InsideGPT`

1.  Install the dependencies using Poetry:



`poetry install`

1.  Run the following command to start the Streamlit app:



`streamlit run main.py`

You should now be able to access the app at `http://localhost:8501` in your web browser.

Using the App
-------------

The user interface of the app is simple and straightforward. You can type in a prompt and generate text based on that prompt. You can also adjust the settings for the language model, such as the maximum length of the generated text and the temperature of the output.

Try it out and see what kind of interesting text you can generate with OpenAI's GPT-3 language model!

Built With
----------

-   [Python](https://www.python.org/) - The programming language used
-   [Streamlit](https://docs.streamlit.io/) - The web framework used for building the app
-   [Poetry](https://python-poetry.org/) - The package manager used for managing dependencies
-   [OpenAI GPT-3](https://beta.openai.com/docs/models/gpt-3) - The language model used for generating text

Contributing
------------

We welcome contributions to the project. If you would like to contribute, please follow the guidelines in the [CONTRIBUTING.md](https://github.com/RemiPelloux/InsideGPT/blob/master/CONTRIBUTING.md) file.

License
-------

This project is licensed under the [MIT License](https://github.com/RemiPelloux/InsideGPT/blob/master/LICENSE).

Acknowledgments
---------------

-   OpenAI for creating the amazing GPT-3 language model
-   The Streamlit team for creating a great framework for building data apps


if poetry doesnt work

Install required packages using pip

-   `pip install streamlit`
-   `pip install openai`
-   `pip install tenacity`
-   `pip install pydantic`
-   `pip install langchain`
-   `pip install faiss-cpu`
-   `pip install embeddings`
-   `pip install docx2txt`
-   `pip install pypdf2`
-   ` pip install python-pptx `
