# Install Miniforge and common Python packages

This document provides step‑by‑step instructions for installing **Miniforge** (a lightweight conda distribution that defaults to `conda-forge`) and then creating an environment and installing common Python packages such as `numpy`, `scipy`, `matplotlib`, `pandas`, `plotly`, `pyside6`, and `jupyter` (Notebook). Commands are shown for **Windows (PowerShell)**, **macOS**, and **Linux**. Where platform differences exist the commands are grouped.

---

## 1. Overview and recommendations

- **Why Miniforge?** Miniforge is a minimal conda distribution that uses the `conda-forge` community channel by default. It is lightweight and widely used in research and reproducible workflows.

---

## 2. Prerequisites

- Internet connection for downloads.
- On Windows: administrative rights are not required for user installs; prefer installing for ``Just Me`` unless you manage multiple accounts.
- On macOS (Apple Silicon) choose the `arm64` installer if you have an M1/M2; choose `x86_64` for Intel macs.

---

## 3. Download & install Miniforge

### 3.1 Windows (PowerShell)

1. Go to https://conda-forge.org/download/ to download the proper version of Miniforge installer according to your OS system.
2. Assume that the installer is saved in the folder of Downloads under the user folder.

**Using `winget` (Windows 10/11 with winget):**


**Manual (if winget not available):**

1. Run the `.exe` and follow the installer prompts. Recommended choices:
   - Install for `Just Me` (unless you have a reason to do system-wide).
   - Allow the installer to initialize `conda` by adding it to the PATH or use the installer default that adds a “conda init” step to your shell.
After install, open a new PowerShell window (or run `conda init powershell` and restart shell).
or open the miniForge Prompt window

### 3.2 macOS

1. Download the correct Miniforge installer from the Miniforge GitHub releases page. Choose `Miniforge3-MacOSX-arm64.sh` for Apple Silicon (M1/M2/M3) or `Miniforge3-MacOSX-x86_64.sh` for Intel.
2. Install from Terminal:

```bash
# Example for arm64 installer (adjust filename if different):
bash ~/Downloads/Miniforge3-MacOSX-arm64.sh

# follow prompts: accept license, and choose install location (default is ~/.miniforge)
# follow the prompts; accept the step of initialization
```
Then close and reopen your Terminal.


### 3.3 Linux

1. Download the appropriate `Miniforge3-Linux-*.sh` installer from the Miniforge releases page.
2. Install from a terminal:

```bash
# example (adjust filename):
bash ~/Downloads/Miniforge3-Linux-x86_64.sh

# follow the prompts; accept the license and choose install path (default ~/.miniforge)
# follow the prompts; accept the step of initialization
```

Re-open the terminal once completed.

---
open the terminal in any OS, you should see the prompt like: (base)xxxx: 

## 4. Verify installation

Open a new shell and run:

```bash
conda --version
```

## 5. Install packages

We will install packages from `conda-forge` where possible because `conda-forge` often has high-quality builds and handles compiled dependencies.

### 6.1 Add `conda-forge` channel (if not already default)

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

> Note: Miniforge defaults to `conda-forge`, but running these commands ensures the configuration in case of a different setup.


This will install all listed packages from `conda-forge`.

### 6.2 Using `conda` (if you prefer not to install mamba)

```bash
conda activate pyenv
conda install numpy scipy matplotlib pandas plotly pyside6 jupyter -c conda-forge -y
```

### 6.4 When to use `pip`

- Use `pip` only when a package is not available (or up-to-date) on `conda-forge`.
- Install packages with `pip` **after** installing conda packages in the active environment.

```bash
# example: install a package via pip within the active conda env
conda activate pyenv
pip install some_package_not_on_conda
```
**Caveat:** mixing `conda` and `pip` is acceptable but can sometimes create conflicts — prefer `conda-forge` when possible.

---

## 7. Install and run Jupyter Notebook

```bash
# install and run
conda install jupyter -c conda-forge -y
jupyter notebook
```

This opens a browser window with the Notebook interface. To stop the server, press `Ctrl+C` in the terminal.
If you prefer JupyterLab:

```bash
mamba install jupyterlab -c conda-forge -y
jupyter lab
```

## 8. Install Inspy and TasVisAn
### 1. Install Inspy
It is recommended to install Inspy first because TasVisAn depend on Inspy while Inspy does not depend on TasVisAn.
1. Download the source code of the InsPy package from github [https://github.com/gcdengansto/inspy/]
2. Unzip the package into a path for keeping source code. e.g. C:\Users\marktwain\mycode\inspy
3. Open terminal and change path to the folder of inspy with the setup.py file
4. Run the following:
```bash
pip install -e .       #the dot is important, meaning the current folder
```
There are some more packages which will be downloaded and installed depending how many has been installed. 
At the end, you should see the last prompt saying that the package is succesfully installed. 

### 2. Install TasVisAn
After installing InsPy, we can follow the same way to install TasVisAn.
1. Download the source code of the TasVisAn package from github [https://github.com/gcdengansto/tasvisan/]
2. Unzip the package into a path for keeping source code. e.g. C:\Users\marktwain\mycode\TasVisAn
3. Open terminal and change path to the folder of inspy with the setup.py file
4. Run the following:
```bash
pip install -e .       #the dot is important, meaning the current folder

```
There are some more packages which will be downloaded and installed depending how many has been installed. 
At the end, you should see the last prompt saying that the package is succesfully installed. 


## 9. Verifying installed packages

After the two packages are successfully installed.
You can verify if they work properly.
Open a terminal to run python:
```bash
(base) C:\Users\marktwain>python
# in the python prompt, import inspy gui interface
>>>import inspy.gui as gui
>>>gui.main_gui.main()

```
For TasVisAn:
```bash
(base) C:\Users\marktwain>python
# in the python prompt, import inspy gui interface
>>>import tasvisan.gui.TASDataBrowser as browser
>>>browser.main()
```
you should see a data browser dialog showing up.


## 10. Common troubleshooting

- **`conda` command not found:** ensure you restarted the shell after installation or manually add the Miniforge `bin`/`Scripts` path to `PATH`.
- **Solver very slow:** install `mamba` and use it instead of `conda`.
- **Conflicting packages:** try creating a fresh environment and avoid mixing many package channels besides `conda-forge`.
- **GUI apps not showing on macOS (PySide6):** for macOS, ensure you have the correct `python.app` integration; launching from Terminal with the `python` from the conda env normally works. In some cases, `pythonw` or `python -m PySide6 ...` may help.

---

## 11. Uninstalling Miniforge

If you need to remove Miniforge:

- Remove the installation directory (for default installs): `~/.miniforge` on macOS/Linux, or `C:\Users\<you>\Miniforge3` on Windows.
- Remove shell initialization lines that `conda init` added to your shell rc files (e.g., `~/.bashrc`, `~/.zshrc`, or PowerShell profile).

---

## 12. Quick command summary

---

## 13. Next steps and variants

---


