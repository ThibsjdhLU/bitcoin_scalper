import os
import ast
import shutil
import sys
from pathlib import Path
from typing import Set, List, Dict

# === CONFIGURATION ===
# Les points d'entrée vitaux (Le Bot + L'Entraînement)
ENTRY_POINTS = [
    "src/bitcoin_scalper/engine_main.py",
    "scripts/train.py",
    "src/bitcoin_scalper/run_dashboard.py" # On garde le training pour ne pas casser la R&D
]

# Dossiers à ignorer totalement (ne pas toucher)
IGNORE_DIRS = [
    "tests", 
    "venv", 
    ".git", 
    "__pycache__"
]

# Destination des fichiers morts
LEGACY_DIR = "src/bitcoin_scalper/legacy"

class DependencyMapper:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir).resolve()
        self.src_root = self.root / "src"
        self.reachable_files: Set[Path] = set()
        self.all_py_files: Set[Path] = set()
        
    def find_all_python_files(self):
        """Liste tous les fichiers .py dans src/"""
        for path in self.src_root.rglob("*.py"):
            # Ignorer les dossiers exclus
            if any(ignore in str(path) for ignore in IGNORE_DIRS):
                continue
            self.all_py_files.add(path)

    def get_imports_from_file(self, file_path: Path) -> Set[str]:
        """Extrait les imports d'un fichier via AST."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except Exception as e:
            print(f"⚠️ Erreur parsing {file_path.relative_to(self.root)}: {e}")
            return set()

        imports = set()
        for node in ast.walk(tree):
            # import x.y
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            # from x.y import z
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
                    # Gérer aussi les imports relatifs (ex: from .core import x)
                    if node.level > 0:
                        # C'est complexe à résoudre parfaitement sans contexte, 
                        # mais on peut souvent ignorer si on scanne tout
                        pass
        return imports

    def resolve_import_to_path(self, import_name: str) -> Path:
        """Convertit 'bitcoin_scalper.core.engine' en chemin de fichier."""
        # Cas 1: Import direct du package
        parts = import_name.split(".")
        
        # Essayer chemin relatif à src/
        potential_path = self.src_root.joinpath(*parts).with_suffix(".py")
        if potential_path.exists():
            return potential_path
            
        # Essayer chemin dossier (pour __init__.py)
        potential_init = self.src_root.joinpath(*parts) / "__init__.py"
        if potential_init.exists():
            return potential_init

        return None

    def build_dependency_graph(self):
        """Parcours en largeur (BFS) pour trouver tout le code vivant."""
        queue = []
        
        # Initialisation avec les Entry Points
        for ep in ENTRY_POINTS:
            p = (self.root / ep).resolve()
            if p.exists():
                queue.append(p)
                self.reachable_files.add(p)
            else:
                print(f"❌ Point d'entrée introuvable : {ep}")

        print(f"🔍 Scan des dépendances depuis {len(queue)} points d'entrée...")
        
        visited = set(queue)
        
        while queue:
            current_file = queue.pop(0)
            
            # Récupérer les imports bruts
            raw_imports = self.get_imports_from_file(current_file)
            
            for imp in raw_imports:
                resolved_path = self.resolve_import_to_path(imp)
                
                # Si l'import correspond à un fichier de notre projet qu'on n'a pas encore vu
                if resolved_path and resolved_path not in visited:
                    # Vérifier si le fichier est bien dans notre scope src/
                    if self.src_root in resolved_path.parents or resolved_path == self.src_root:
                        visited.add(resolved_path)
                        self.reachable_files.add(resolved_path)
                        queue.append(resolved_path)

    def move_zombies(self, dry_run: bool = True):
        """Déplace les fichiers non atteints."""
        # Les zombies sont tous les fichiers .py de src/ moins ceux qu'on a atteints
        # On exclut aussi __init__.py racine s'il existe pour ne pas casser le package
        zombies = self.all_py_files - self.reachable_files
        
        # Filtrage fin : on garde toujours le __init__.py principal du package
        pkg_init = self.src_root / "bitcoin_scalper" / "__init__.py"
        if pkg_init in zombies:
            zombies.remove(pkg_init)

        print(f"\n📊 BILAN :")
        print(f"   - Fichiers vivants : {len(self.reachable_files)}")
        print(f"   - Fichiers ZOMBIES (Inutilisés) : {len(zombies)}")
        
        if not zombies:
            print("✅ Aucun fichier mort détecté. Le projet est propre !")
            return

        print("\n💀 LISTE DES ZOMBIES :")
        for z in sorted(zombies):
            rel_path = z.relative_to(self.root)
            print(f"   - {rel_path}")

        if dry_run:
            print("\n🛑 MODE DRY RUN : Aucune modification effectuée.")
            print("   Lancez 'python3 cleanup_repo.py' (sans arguments) pour exécuter le déplacement.")
            return

        # Création du dossier Legacy
        legacy_path = self.root / LEGACY_DIR
        legacy_path.mkdir(parents=True, exist_ok=True)
        (legacy_path / "__init__.py").touch()
        
        print(f"\n🚜 Déplacement vers {LEGACY_DIR}...")
        
        for z in zombies:
            # Calculer le chemin relatif pour garder la structure
            # ex: src/bitcoin_scalper/old/stuff.py -> legacy/old/stuff.py
            try:
                rel_path = z.relative_to(self.src_root / "bitcoin_scalper")
            except ValueError:
                # Fichier hors du package principal
                rel_path = z.relative_to(self.src_root)
                
            dest = legacy_path / rel_path
            
            # Créer les dossiers parents
            dest.parent.mkdir(parents=True, exist_ok=True)
            
            # Déplacer
            shutil.move(str(z), str(dest))
            print(f"   ✓ Déplacé : {z.name}")

        print("\n✨ Nettoyage terminé !")

if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    
    mapper = DependencyMapper(".")
    mapper.find_all_python_files()
    mapper.build_dependency_graph()
    mapper.move_zombies(dry_run=dry_run)
