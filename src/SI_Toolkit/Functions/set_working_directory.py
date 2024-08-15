"""
Not used currently
"""

import os


def set_working_directory():
    """
    Ensures that the working directory is set so that 'SI_Toolkit_ASF' is directly in it.

    This function checks if the current working directory (cwd) includes 'SI_Toolkit_ASF'.
    If 'SI_Toolkit_ASF' is not in cwd, the function ascends the directory tree,
    looking for a directory that includes 'SI_Toolkit_ASF'. If such a directory is found,
    the cwd is changed to this directory. The goal is to ensure that operations depending
    on the relative location of 'SI_Toolkit_ASF' can proceed correctly.

    Affects the operating environment by potentially changing the current working directory.

    Prints messages indicating the outcome of its operations.
    """
    # We know that we want target directory to be directly in the working directory
    target_directory = "SI_Toolkit_ASF"
    # Get the absolute path of the current working directory
    current_dir = os.path.abspath(os.getcwd())

    # Keep track of the previous directory to prevent infinite loops.
    # This is a safeguard in case the target directory isn't found.
    prev_dir = None

    # Continue moving up the directory hierarchy until the target directory is found in the current directory,
    # or until the current directory does not change
    # (which would mean we've reached the root directory without finding the target).
    while target_directory not in os.listdir(current_dir) and current_dir != prev_dir:
        # Update prev_dir to the current directory before moving up
        prev_dir = current_dir
        # Move up one directory level
        current_dir = os.path.dirname(current_dir)

    # After exiting the loop, check if the target directory is in the current directory.
    # This indicates that we've found the right directory in the hierarchy.
    if target_directory in os.listdir(current_dir) and os.path.isdir(os.path.join(current_dir, target_directory)):
        # Change the working directory to the found target directory.
        os.chdir(current_dir)
        print(f"Working directory set to: {current_dir}")
    else:
        # If the target directory was not found, it's likely it doesn't exist in the path.
        # This is an error state, and the user should be informed.
        print("Target directory not found in the path hierarchy. Please check the directory structure.")
