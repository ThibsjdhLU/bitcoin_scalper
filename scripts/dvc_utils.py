#!/usr/bin/env python3
"""
Script utilitaire CLI pour DVC (init, add, commit, push, pull, repro, status, diff, gc, remote).
Usage : python scripts/dvc_utils.py <commande> [options]
"""
import argparse
from app.core.dvc_manager import DVCManager


def main():
    parser = argparse.ArgumentParser(description="Outils DVC pipeline (wrapper Python)")
    parser.add_argument("cmd", choices=[
        "init", "add", "commit", "push", "pull", "repro", "status", "diff", "gc", "remote"
    ], help="Commande DVC à exécuter")
    parser.add_argument("--path", help="Chemin fichier/dossier pour add/commit")
    parser.add_argument("--target", help="Cible pour repro")
    parser.add_argument("--workspace", action="store_true", help="GC sur workspace uniquement")
    parser.add_argument("--remote_action", help="Action remote (add/remove/modify)")
    parser.add_argument("--remote_name", help="Nom du remote")
    parser.add_argument("--remote_url", help="URL du remote")
    parser.add_argument("--repo", help="Chemin du repo DVC")
    args = parser.parse_args()

    dvc = DVCManager(repo_path=args.repo)

    if args.cmd == "init":
        print("DVC init:", dvc.init())
    elif args.cmd == "add":
        print("DVC add:", dvc.add(args.path))
    elif args.cmd == "commit":
        print("DVC commit:", dvc.commit(args.path))
    elif args.cmd == "push":
        print("DVC push:", dvc.push())
    elif args.cmd == "pull":
        print("DVC pull:", dvc.pull())
    elif args.cmd == "repro":
        print("DVC repro:", dvc.repro(args.target))
    elif args.cmd == "status":
        print(dvc.status())
    elif args.cmd == "diff":
        print(dvc.diff())
    elif args.cmd == "gc":
        print("DVC gc:", dvc.gc(workspace=args.workspace))
    elif args.cmd == "remote":
        print("DVC remote:", dvc.remote(args.remote_action, args.remote_name, args.remote_url))
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 