**1. Project Initialization & Core Dependencies (Cross-Platform)**

* `uv init --python 3.12`
* `uv add kaggle split-folders albumentations opencv-python-headless matplotlib scikit-learn`

**2. Configure Cross-Platform GPU Markers**
Append the environment markers directly to your `pyproject.toml` via the terminal so `uv` knows how to route PyTorch downloads based on the active operating system.

* **macOS:**

```bash
cat << 'EOF' >> pyproject.toml

[tool.uv.sources]
torch = [{ index = "pytorch-cu121", marker = "sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cu121", marker = "sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
EOF

```

* **Windows (PowerShell):**

```powershell
@"

[tool.uv.sources]
torch = [{ index = "pytorch-cu121", marker = "sys_platform == 'win32'" }]
torchvision = [{ index = "pytorch-cu121", marker = "sys_platform == 'win32'" }]

[[tool.uv.index]]
name = "pytorch-cu121"
url = "https://download.pytorch.org/whl/cu121"
explicit = true
"@ | Add-Content -Path pyproject.toml

```

**3. Universal PyTorch Installation (Cross-Platform)**
Because the OS markers are configured, you only need to run this identical command on either machine. `uv` will automatically grab the `cu121` wheel on Windows and the standard Apple-optimized wheel on Mac.

* `uv add torch torchvision`

**4. Kaggle Authentication**

* **macOS:**

```bash
mkdir -p ~/.kaggle && echo your-api-key > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token

```

* **Windows (PowerShell):**

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.kaggle"
Set-Content -Path "$env:USERPROFILE\.kaggle\access_token" -Value "your-api-key"

```

**5. Dataset Download & Extraction**

* **Download (Cross-Platform):**
`uv run kaggle datasets download -d tawsifurrahman/covid19-radiography-database`
* **Extract (macOS):**
`unzip covid19-radiography-database.zip -d raw_data`
* **Extract (Windows PowerShell):**
    * `New-Item -ItemType Directory -Name "raw_data"`
    * `tar -xf covid19-radiography-database.zip -C raw_data`

**6. Run Data Pipeline Scripts (Cross-Platform)**

* `uv run src/split_data.py`
* `uv run src/data.py`
* `uv run src/visualize_precprocess.py`

**7. Run Training Scripts (Cross-Platform)**

* `uv run src/train.py`