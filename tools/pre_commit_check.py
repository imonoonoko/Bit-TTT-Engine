import os
import sys

# Limit: 99MB (GitHub limit is 100MB, but safety margin is good)
MAX_SIZE_BYTES = 99 * 1024 * 1024

def check_large_files():
    root_dir = os.getcwd()
    large_files = []

    print("[Pre-Commit Check] Scanning for large files (>99MB)...")

    # Walk through all files
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip .git and target directories explicitly to save time
        if ".git" in dirnames:
            dirnames.remove(".git")
        if "target" in dirnames:
            dirnames.remove("target")
        
        for name in filenames:
            file_path = os.path.join(dirpath, name)
            
            # Skip if it's not a file (symlink etc)
            if not os.path.isfile(file_path):
                continue
                
            try:
                size = os.path.getsize(file_path)
                if size > MAX_SIZE_BYTES:
                    # Check if ignored? 
                    # Naive check: just report it. 
                    # If it's ignored by .gitignore, usually git won't commit it, 
                    # BUT if it was force-added or tracked, this hook should catch it.
                    # Ideally we only check *staged* files, but that requires git command interaction.
                    # For a robust "Repo Health" check, finding ANY large file in the tree that isn't excluded 
                    # by script logic is handled here, but hooks usually only care about what's being committed.
                    
                    # Optimization: We should really only check staged files if running as a hook.
                    # But for now, let's make this a general scanner.
                    # If this script is run by the hook, we might want to check `git diff --cached --name-only`
                    pass
            except OSError:
                pass

    # Better approach for Pre-Commit: Use git to find staged files and check THEIR sizes.
    import subprocess
    try:
        # List staged files
        result = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True, check=True)
        staged_files = result.stdout.splitlines()
        
        for rel_path in staged_files:
            if not rel_path: continue
            full_path = os.path.join(root_dir, rel_path)
            if os.path.exists(full_path):
                size = os.path.getsize(full_path)
                if size > MAX_SIZE_BYTES:
                    large_files.append((rel_path, size / (1024*1024)))
            else:
                # Deleted file
                pass
                
    except subprocess.CalledProcessError:
        print("Warning: Not a git repository or git error.")
        return 0

    if large_files:
        print("❌ ERROR: Large files detected in staging area!")
        for path, size_mb in large_files:
            print(f"   - {path} ({size_mb:.2f} MB)")
        print("GitHub rejects files > 100MB.")
        print("Please remove these files from the commit (git reset HEAD <file>) or use Git LFS.")
        return 1
    
    print("✅ No large files found in staging area.")
    return 0

if __name__ == "__main__":
    sys.exit(check_large_files())
