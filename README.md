# Invoice

## Prerequisites

Before you begin, ensure you have Python installed on your system. 

## Developer Setup

This project includes a `setup.sh` script to automate the setup process.

1.  **Clone the repository:**
    ```bash
    git clone git@github.com:ryanznie/invoice.git
    cd invoice
    ```

2.  **Run the setup script:**
    This will create a virtual environment, install all necessary dependencies using `uv`, and set up pre-commit hooks.
    ```bash
    bash setup.sh
    ```

3.  **Activate the virtual environment:**
    ```bash
    source .venv/bin/activate
    ```