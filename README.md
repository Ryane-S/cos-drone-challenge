# Challenge COS - Autoguidage laser d'un drone autonome

Projet de fin d'études de la VAP HTI - Télécom SudParis.

Ce projet porte sur l'autoguidage laser d'un drone autonome. Il couvre le fine-tuning
et l'inférence sur des architectures de segmentation d'images, dans le cadre d'un
challenge académique.

## Installation

### 1. Installer `uv`

**Windows (PowerShell)**

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Linux / macOS**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Cloner le dépôt

```bash
git clone https://github.com/Ryane-S/cos-drone-challenge.git
```

### 3. Installer les dépendances

La commande suivante crée automatiquement un environnement virtuel et installe les dépendances spécifiées dans `pyproject.toml` et `uv.lock`.

```bash
uv sync
```
