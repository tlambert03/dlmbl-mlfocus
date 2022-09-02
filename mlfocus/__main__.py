if __name__ == "__main__":
    import sys
    from pathlib import Path
    from mlfocus.util import folder_to_patches

    root = Path(sys.argv[1])

    for subd in root.iterdir():
        if subd.is_dir():
            folder_to_patches(subd)
