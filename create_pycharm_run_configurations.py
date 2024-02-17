import os
import shutil


def create_run_configurations(templates_directory):
    idea_dir = '../.idea'
    run_configs_dir = os.path.join(idea_dir, 'runConfigurations')

    if not os.path.exists(idea_dir):
        print(f"The .idea directory does not exist in {idea_dir}.")
        return

    if not os.path.exists(run_configs_dir):
        os.makedirs(run_configs_dir)
        print(f"Created runConfigurations directory in {idea_dir}.")

    for filename in os.listdir(templates_directory):
        if filename.endswith('.xml.template'):
            dest_filename = filename.replace('.template', '')
            dest_file = os.path.join(run_configs_dir, dest_filename)

            # Check if the configuration already exists
            if not os.path.exists(dest_file):
                src_file = os.path.join(templates_directory, filename)
                shutil.copy(src_file, dest_file)
                print(f"Copied {filename} to {dest_file}")
            else:
                print(f"Configuration {dest_filename} already exists. Skipping.")
