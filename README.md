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

## Lancement script de prédiction

Pour lancer la prédiction il est essentiel de se prémunir d'un **token Sam3**. Pour cela il est nécessaire :
1. D'obtenir l'autorisation d'accès au modèle sam3 sur [HuggingFace](https://huggingface.co/facebook/sam3).
2. D'accéder à ses tokens via [Access Tokens](https://huggingface.co/settings/tokens) puis en générer un.
3. De saisir le token dans le fichier `sam3_token` à la racine du dépôt ou via la commande suivante :

```bash
echo "[VOTRE_TOKEN]" > ./sam3_token
```

Il ne reste plus qu'à exécuter le script `sam_3_predictor.py` via la commande :

```bash
uv run python -m models.sam3.sam_3_predictor
```
